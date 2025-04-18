import scipy.sparse as sp
import numpy as np


# Function to split the data into nfold cross-validation sets
def splitData(dataY, splitPath, nfold, seed_cross, crossKey='cross'):
    neg_pos_ratio = 1  # The ratio of negative to positive samples, here it's 1 (equal number of negative and positive samples)

    # Find the indices of positive and negative samples in the dataY matrix
    index_pos = np.array(np.where(dataY == 1))  # Indices of positive samples
    index_neg = np.array(np.where(dataY == 0))  # Indices of negative samples

    pos_num = len(index_pos[0])  # Number of positive samples
    neg_num = int(pos_num * neg_pos_ratio)  # Number of negative samples based on the neg_pos_ratio

    # Shuffle the positive and negative samples using the given seed
    np.random.seed(seed_cross)
    np.random.shuffle(index_pos.T)
    np.random.seed(seed_cross)
    np.random.shuffle(index_neg.T)

    # Select a subset of negative samples to match the number of positive samples (according to the ratio)
    index_neg = index_neg[:, : neg_num]

    # Create a fold index for positive and negative samples to distribute them across the folds
    cross_fold_index_pos = np.array(
        [temp % nfold for temp in range(len(index_pos[0]))])  # Assign each positive sample to a fold
    cross_fold_index_neg = np.array(
        [temp % nfold for temp in range(len(index_neg[0]))])  # Assign each negative sample to a fold

    # Loop through each fold
    for kfold in range(nfold):
        # Split the data into training and testing sets for this fold
        cross_tra_fold_pos = index_pos.T[cross_fold_index_pos != kfold]
        cross_tes_fold_pos = index_pos.T[cross_fold_index_pos == kfold]
        cross_tra_fold_neg = index_neg.T[cross_fold_index_neg != kfold]
        cross_tes_fold_neg = index_neg.T[cross_fold_index_neg == kfold]

        # Combine the positive and negative samples for training and testing
        cross_tra_fold = np.vstack((cross_tra_fold_pos, cross_tra_fold_neg))
        cross_tes_fold = np.vstack((cross_tes_fold_pos, cross_tes_fold_neg))

        # Create the training and testing data, adding labels to each sample
        cross_tra_data = np.hstack((cross_tra_fold, dataY[cross_tra_fold[:, 0], cross_tra_fold[:, 1]].reshape(-1, 1)))
        cross_tes_data = np.hstack((cross_tes_fold, dataY[cross_tes_fold[:, 0], cross_tes_fold[:, 1]].reshape(-1, 1)))

        # Convert the data into sparse matrices (COO format) for efficient storage and computation
        cross_tra_matx = sp.coo_matrix((cross_tra_data[:, 2], (cross_tra_data[:, 0], cross_tra_data[:, 1])),
                                       shape=(dataY.shape[0], dataY.shape[1])).toarray()
        cross_tes_matx = sp.coo_matrix((cross_tes_data[:, 2], (cross_tes_data[:, 0], cross_tes_data[:, 1])),
                                       shape=(dataY.shape[0], dataY.shape[1])).toarray()

        # Save the training and testing data for this fold to text files
        np.savetxt(splitPath + crossKey + '_tra_kfold' + str(kfold) + '_seed' + str(seed_cross) + '.txt',
                   cross_tra_data, fmt='%d', delimiter=',')
        np.savetxt(splitPath + crossKey + '_tes_kfold' + str(kfold) + '_seed' + str(seed_cross) + '.txt',
                   cross_tes_data, fmt='%d', delimiter=',')

        # Extract only the positive samples from the training and testing data
        cross_tra_data_total = cross_tra_data[cross_tra_data[:, -1] == 1][:, :-1]
        cross_tes_data_total = cross_tes_data[cross_tes_data[:, -1] == 1][:, :-1]

        # Adjust indices for the total dataset (by adding the number of rows in dataY)
        cross_tra_data_total[:, 1] += dataY.shape[0]
        cross_tes_data_total[:, 1] += dataY.shape[0]

        # Save the positive samples for training and testing to separate files
        np.savetxt(splitPath + crossKey + '_tra_kfold' + str(kfold) + '_seed' + str(seed_cross) + '_total.txt',
                   cross_tra_data_total, fmt='%d', delimiter=' ')
        np.savetxt(splitPath + crossKey + '_tes_kfold' + str(kfold) + '_seed' + str(seed_cross) + '_total.txt',
                   cross_tes_data_total, fmt='%d', delimiter=' ')

    return  # Return nothing, as the data is saved to files


# Main function to execute the data splitting process
if __name__ == "__main__":
    nfold = 5  # Number of folds for cross-validation
    seed_cross = 1  # Random seed for reproducibility

    DATASET = 'LuoDTI'  # Dataset name
    dataPath = '../dataset/' + DATASET + '/'  # Path to the dataset
    usedDataPath = dataPath  # Path to the used data
    splitPath = dataPath + 'input/'  # Path to save the split data

    # Load the dataset (assuming it's a text file with space-separated values)
    dataY = np.genfromtxt(usedDataPath + 'mat_drug_protein.txt', dtype=float, delimiter=' ')

    # Call the function to split the data into nfolds and save it
    splitData(dataY, splitPath, nfold, seed_cross)
