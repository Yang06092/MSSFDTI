import pickle
import timeit
import numpy as np
import random
import logging
import torch
import torch.utils.data as data_utils
from collections import defaultdict
from torch.optim import Optimizer, SGD
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.functional import Tensor
from torch.nn.functional import softmax
from sklearn.metrics import accuracy_score
from .model.MSSFDTI import DTI
from pretrain_code.dataprocessing import preprocess_adj, normalize_features
from pretrain_code.torch_data import read_txt, read_csv
from .metrics import get_metrics

# Function to extract non-zero elements' edge indices from the matrix
def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return Tensor(edge_index)


# Lookahead optimizer implementation
class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k  # Number of steps for updating the "slow" parameters
        self.alpha = alpha  # Update rate of slow parameters
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0  # Track the number of steps since last update

    def update(self, group):
        # Update the slow parameters with the fast parameters
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        # Take a step for optimization and update the slow parameters
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        # Save the state dictionaries for both fast and slow parameters
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        # Load the state dictionaries for both fast and slow parameters
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        # Add a new parameter group to the optimizer
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)


# Function to load tensor from a file
def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy')]


# Function to load pickle file
def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


# Set the random seed for reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Shuffle dataset randomly
def random_shuffle(dataset, seed):
    random.seed(seed)
    random.shuffle(dataset)
    return dataset


# Trainer class for training the model
class Trainer(object):
    def __init__(self, model, lr=0.1, weight_decay=5e-4):
        self.model = model
        self.optimizer_inner = SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)

    def train(self, train_dataset, datasetF, batch_size=32):
        train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        criterion = CrossEntropyLoss().to(device)
        train_labels = []
        train_preds = []
        total_loss = 0
        total_batches = len(train_loader)

        all_train_drug_covector = []
        all_train_protein_covector = []
        all_train_labels = []
        predicted_train = []

        for batch in train_loader:
            self.optimizer.zero_grad()

            train_drug_covector, train_protein_covector, predicted, interaction = self.model(batch, datasetF)
            loss = criterion(predicted, interaction)
            loss.backward()

            clip_grad_norm_(parameters=self.model.parameters(), max_norm=5)
            self.optimizer.step()

            total_loss += loss.item()

            preds = predicted.max(1)[1]
            train_labels.extend(interaction.cpu().detach().numpy())
            train_preds.extend(preds.cpu().detach().numpy())

            all_train_drug_covector.append(train_drug_covector)
            all_train_protein_covector.append(train_protein_covector)
            all_train_labels.append(interaction)
            predicted_train.append(softmax(predicted, dim=-1)[:, 1].detach().cpu())

        all_train_drug_covector = torch.cat(all_train_drug_covector, dim=0)
        all_train_protein_covector = torch.cat(all_train_protein_covector, dim=0)
        all_train_labels = torch.cat(all_train_labels, dim=0)
        predicted_train = torch.cat(predicted_train, dim=0)

        train_acc = accuracy_score(train_labels, train_preds)
        avg_loss = total_loss / total_batches

        return avg_loss, train_acc, all_train_drug_covector, all_train_protein_covector, all_train_labels, predicted_train


# Tester class for testing and evaluation
class Tester(object):
    def __init__(self, model):
        self.model = model.to(device)

    def dev(self, epoch, dev_dataset, datasetF, batch_size=32):
        test_loader = data_utils.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

        test_labels = []
        test_preds = []
        test_scores = []

        all_dev_drug_covector = []
        all_dev_protein_covector = []
        all_test_labels = []

        with torch.no_grad():
            for batch in test_loader:
                dev_drug_covector, dev_protein_covector, predicted, interaction = self.model(batch, datasetF)

                ys = softmax(predicted, 1).cpu().detach().numpy()
                predicted_labels = list(map(lambda x: np.argmax(x), ys))
                predicted_scores = list(map(lambda x: x[1], ys))

                test_labels.extend(interaction.cpu().numpy())
                test_preds.extend(predicted_labels)
                test_scores.extend(predicted_scores)

                all_dev_drug_covector.append(dev_drug_covector)
                all_dev_protein_covector.append(dev_protein_covector)
                all_test_labels.append(interaction)

            all_dev_drug_covector = torch.cat(all_dev_drug_covector, dim=0)
            all_dev_protein_covector = torch.cat(all_dev_protein_covector, dim=0)
            all_test_labels = torch.cat(all_test_labels, dim=0)

            metrics, metric_names = get_metrics(np.array(test_labels), np.array(test_scores))

            test_auc = metrics[2]
            test_aupr = metrics[3]
            f1 = metrics[4]
            accuracy = metrics[5]
            recall = metrics[6]
            specificity = metrics[7]
            precision = metrics[8]

            return test_labels, test_scores, accuracy, test_auc, test_aupr, precision, recall, f1, specificity, all_dev_drug_covector, all_dev_protein_covector

    def save_results(self, result_data, filename):
        result_data = [item.cpu().numpy() if isinstance(item, torch.Tensor) else item for item in result_data]
        np.save(filename, np.array(result_data, dtype=object))

    def test(self, epoch, test_dataset, datasetF, batch_size=32):
        test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        test_labels = []
        test_preds = []
        test_scores = []

        all_test_drug_covector = []
        all_test_protein_covector = []
        all_test_labels = []

        with torch.no_grad():
            for batch in test_loader:
                test_drug_covector, test_protein_covector, predicted, interaction = self.model(batch, datasetF)

                ys = softmax(predicted, 1).cpu().detach().numpy()
                predicted_labels = list(map(lambda x: np.argmax(x), ys))
                predicted_scores = list(map(lambda x: x[1], ys))

                test_labels.extend(interaction.cpu().numpy())
                test_preds.extend(predicted_labels)
                test_scores.extend(predicted_scores)

                all_test_drug_covector.append(test_drug_covector)
                all_test_protein_covector.append(test_protein_covector)
                all_test_labels.append(interaction)

        all_test_drug_covector = torch.cat(all_test_drug_covector, dim=0)
        all_test_protein_covector = torch.cat(all_test_protein_covector, dim=0)
        all_test_labels = torch.cat(all_test_labels, dim=0)

        return test_labels, test_scores, all_test_drug_covector, all_test_protein_covector

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a+') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


# The main function to handle cross-validation and model training/evaluation
def train(DATASET, fold, save_auc, attention, random_seed, log_write=True):
    result_data = []  # Stores the results of each fold

    global best_train_acc, best_acc, best_epoch, best_aupr, best_loss_train, best_precision, best_f1, best_specificity, best_recall
    dir_input = ('../dataset/' + DATASET)

    # Load train, dev, and test datasets
    train_dataset = np.loadtxt(dir_input + '/input/cross_tra_kfold{}_seed1.txt'.format(fold), dtype=int, delimiter=',')
    dev_dataset = np.loadtxt(dir_input + '/input/cross_tes_kfold{}_seed1.txt'.format(fold), dtype=int, delimiter=',')
    test_dataset = dev_dataset

    # Read drug-protein matrix and features
    drug_protein_matrix = read_txt(dir_input + '/input/mat_drug_protein.txt', ' ')
    drug_num = len(drug_protein_matrix)
    protein_num = len(drug_protein_matrix[0])

    datasetF = dict()

    # Preprocess drug and protein similarity matrices
    drug_sim_matrix = read_csv(dir_input + '/input/dim/drug_f512_matrix708.csv')
    drug_sim_matrix1 = preprocess_adj(drug_sim_matrix)
    drug_sim_edge_index = get_edge_index(drug_sim_matrix1)

    drug_feature = read_csv(dir_input + '/input/dim/drug_f512_feature708.csv')
    drug_feature = normalize_features(drug_feature)

    datasetF['dd'] = {'matrix': drug_sim_matrix1, 'edges': drug_sim_edge_index, 'feature': drug_feature}

    protein_sim_matrix = read_csv(dir_input + '/input/dim/protein_f512_matrix1512.csv')
    protein_sim_matrix1 = preprocess_adj(protein_sim_matrix)
    protein_sim_edge_index = get_edge_index(protein_sim_matrix1)

    protein_feature = read_csv(dir_input + '/input/dim/protein_f512_feature1512.csv')
    protein_feature = normalize_features(protein_feature)
    datasetF['pp'] = {'matrix': protein_sim_matrix1, 'edges': protein_sim_edge_index, 'feature': protein_feature}

    # Continue processing other matrices and features for drug-protein interaction (dp) and other combinations
    # More data processing code...

    model = DTI(drug_num, protein_num, dim, layer_output, layer_IA, nhead, dropout, attention=attention).to(device)

    trainer = Trainer(model)
    tester = Tester(model)

    start = timeit.default_timer()

    best_auc = 0
    es = 0  # Early stopping counter

    for epoch in range(0, iteration):
        print('---------epoch{}---------'.format(epoch))
        loss_train, _, train_drug_covector, train_protein_covector, train_labels, predicted_train = trainer.train(
            train_dataset, datasetF)

        # Get AUC and other evaluation metrics
        test_labels, test_scores, dev_acc, dev_auc, dev_aupr, dev_precision, dev_recall, dev_f1, dev_specificity, dev_drug_covector, dev_protein_covector = tester.dev(
            epoch, dev_dataset, datasetF)

        AUCs = [epoch, loss_train, dev_acc, dev_auc, dev_aupr, dev_precision, dev_recall, dev_f1, dev_specificity]

        print("Metrics for the current epoch:")
        print('\t'.join(map(str, AUCs)))

        # Save the best model based on dev_auc
        if dev_auc > best_auc:
            best_epoch = epoch
            best_loss_train = loss_train
            best_acc = dev_acc
            best_auc = dev_auc
            best_aupr = dev_aupr
            best_precision = dev_precision
            best_recall = dev_recall
            best_f1 = dev_f1
            best_specificity = dev_specificity

            print('New best performance achieved.')
            model.save_fold(fold, train_drug_covector, train_protein_covector, train_labels, dev_drug_covector,
                            dev_protein_covector, test_labels, predicted_train)
            es = 0

            # Save best model's results
            result_data = [
                test_labels,
                test_scores,
                best_auc,
                best_aupr,
                best_f1,
                best_acc,
                best_recall,
                best_specificity,
                best_precision,
            ]

            result_filename = f"cross_validation_results_fold_{fold}.npy"
            tester.save_results(result_data, result_filename)
            print(f"Best results for fold {fold} saved to {result_filename}")

        # Early stopping mechanism
        if dev_auc <= best_auc:
            es += 1
            if es > 20:
                print('Early stopping counter reached 20, stopping training.')
                break

    return best_acc, best_auc, best_aupr, best_precision, best_recall, best_f1, best_specificity, result_data


# Main execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')

    """Hyperparameters."""
    nfold = 5
    DATASET = 'LuoDTI'
    dim = 512
    layer_IA = 3
    lr = 0.1
    weight_decay = 5e-4
    iteration = 100
    random_seed = 2021
    optimizer = 'lookahead-SGD'
    layer_output = 3
    nhead = 8
    dropout = 0.1
    attention = 'DIA'

    # Convert hyperparameters to correct data types
    (dim, layer_output, layer_IA, iteration, random_seed, nhead) = map(int, [dim, layer_output, layer_IA, iteration,
                                                                             random_seed, nhead])
    lr, weight_decay, dropout = map(float, [lr, weight_decay, dropout])

    # Set up device and model configurations
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
        print('Using GPU')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    result_acc = np.zeros((nfold))
    result_auc = np.zeros((nfold))
    result_aupr = np.zeros((nfold))
    result_pre = np.zeros((nfold))
    result_recall = np.zeros((nfold))
    result_f1 = np.zeros((nfold))
    result_specificity = np.zeros((nfold))

    result_data_all_folds = []  # For storing results of all folds
    for fold in range(0, nfold):
        best_acc, best_auc, best_aupr, best_precision, best_recall, best_f1, best_specificity, result_data = train(
            DATASET, fold, 0.9, attention, random_seed, log_write=True)
        result_data_all_folds.append(result_data)  # Save fold results
        result_acc[fold] = best_acc
        result_auc[fold] = best_auc
        result_aupr[fold] = best_aupr
        result_pre[fold] = best_precision
        result_recall[fold] = best_recall
        result_f1[fold] = best_f1
        result_specificity[fold] = best_specificity

    # Save all fold results to a file
    np.save("cross_validation_results.npy", np.array(result_data_all_folds, dtype=object))  # Shape should be (5, 9)
    print("Cross-validation results saved as 'cross_validation_results.npy'.")

    # Print statistics for the entire cross-validation process
    # Print statistics for the entire cross-validation process
    metrics = {
        'Accuracy': result_acc,
        'AUC': result_auc,
        'AUPR': result_aupr,
        'Precision': result_pre,
        'Recall': result_recall,
        'F1 Score': result_f1,
        'Specificity': result_specificity
    }

    # Print each metric's results and averages
    for metric_name, values in metrics.items():
        print(f'Result {metric_name}s: {values}')
        print(f'Avg {metric_name}: {values.mean():.4f} ± {values.std():.4f}')

