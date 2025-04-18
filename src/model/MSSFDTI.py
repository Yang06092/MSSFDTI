import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math
from MGCN_layer import GCN_homo, GCN_bi, GCN_hete
from MSBF import MSBF
from RESA import RESA

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- DIM (Deep Interaction Mechanism) Module ---
class DIM(nn.Module):
    """
    This class implements the Deep Interaction Mechanism (DIM) which applies RESA (Relation-Aware Self-Attention)
    and MSBF (Multi-Scale Bilateral Fusion) modules for drug-protein interactions.
    """

    def __init__(self, dim, nhead, dropout):
        super(DIM, self).__init__()
        # Initialize RESA and MSBF modules for both drug and protein
        self.dRESA = RESA(dim, nhead, dropout)
        self.tRESA = RESA(dim, nhead, dropout)
        self.dMSBF = MSBF(dim, nhead, dropout)
        self.tMSBF = MSBF(dim, nhead, dropout)

    def forward(self, drug_vector, protein_vector):
        # Apply RESA and MSBF to the drug and protein vectors
        drug_vector = self.dRESA(drug_vector, None)
        protein_vector = self.tRESA(protein_vector, None)
        drug_covector = self.dMSBF(drug_vector, protein_vector, None)
        protein_covector = self.tMSBF(protein_vector, drug_vector, None)
        return drug_covector, protein_covector


# --- MGCN Layer ---
class MGCNLayer(nn.Module):
    """
    This class implements the Multi-Scale Graph Convolutional Layer (MGCN) which utilizes
    both homogeneous and bipartite graph convolutions for drug-protein interactions.
    """

    def __init__(self, input_d, input_p, dim):
        super(MGCNLayer, self).__init__()
        self.input_d = input_d
        self.input_p = input_p

        # Initialize GCN modules for homogeneous and bipartite graphs
        self.gcn_homo_d = GCN_homo(input_d, dim)
        self.gcn_homo_p = GCN_homo(input_p, dim)
        self.gcn_bi = GCN_bi(input_d + input_p, dim)
        self.gcn_hete = GCN_hete(input_d + input_p, dim)

    def forward(self, datasetF):
        # Apply graph convolutions for each type of graph (homogeneous, bipartite, heterogeneous)
        x_d_dr1, x_d_dr2, x_d_dr3 = self.gcn_homo_d(datasetF['dd']['matrix'].to(device),
                                                    datasetF['dd']['feature'].to(device))
        y_p_pro1, y_p_pro2, y_p_pro3 = self.gcn_homo_p(datasetF['pp']['matrix'].to(device),
                                                       datasetF['pp']['feature'].to(device))
        dp1, dp2, dp3 = self.gcn_bi(datasetF['dp']['matrix'].to(device), datasetF['dp']['feature'].to(device))
        ddpp1, ddpp2, ddpp3 = self.gcn_hete(datasetF['ddpp']['matrix'].to(device),
                                            datasetF['ddpp']['feature'].to(device))

        # Extract relevant parts from the convolutions
        x_d_dr4 = dp1[:self.input_d, :]
        y_p_pro4 = dp1[self.input_d:, :]
        x_d_dr5 = dp2[:self.input_d, :]
        y_p_pro5 = dp2[self.input_d:, :]
        x_d_dr6 = dp3[:self.input_d, :]
        y_p_pro6 = dp3[self.input_d:, :]
        x_d_dr7 = ddpp1[:self.input_d, :]
        y_p_pro7 = ddpp1[self.input_d:, :]
        x_d_dr8 = ddpp2[:self.input_d, :]
        y_p_pro8 = ddpp2[self.input_d:, :]
        x_d_dr9 = ddpp3[:self.input_d, :]
        y_p_pro9 = ddpp3[self.input_d:, :]

        # Stack the convoluted features
        x_d_dr = torch.stack((x_d_dr1, x_d_dr2, x_d_dr3, x_d_dr4, x_d_dr5, x_d_dr6, x_d_dr7, x_d_dr8, x_d_dr9), 0)
        y_p_pro = torch.stack(
            (y_p_pro1, y_p_pro2, y_p_pro3, y_p_pro4, y_p_pro5, y_p_pro6, y_p_pro7, y_p_pro8, y_p_pro9), 0)

        # Transpose to match the required shape for further processing
        x_d_dr = torch.transpose(x_d_dr, 0, 1)
        y_p_pro = torch.transpose(y_p_pro, 0, 1)

        return x_d_dr, y_p_pro


# --- DTI (Drug-Target Interaction) Model ---
class DTI(nn.Module):
    """
    The Drug-Target Interaction (DTI) model integrates MGCN and DIM modules for drug-protein interaction prediction.
    """

    def __init__(self, input_d, input_p, dim, layer_output, layer_IA, nhead, dropout, attention):
        super(DTI, self).__init__()

        self.gcnlayer = MGCNLayer(input_d, input_p, dim)  # Initialize the MGCN layer
        self.attention = attention
        self.layer_IA = layer_IA
        self.DIA_ModuleList = nn.ModuleList(
            [DIM(dim, nhead, dropout) for _ in range(layer_IA)])  # Initialize the DIM layers
        self.dr_lin = nn.Linear((layer_IA + 1) * dim, dim)  # Linear transformation for drug vector
        self.pro_lin = nn.Linear((layer_IA + 1) * dim, dim)  # Linear transformation for protein vector
        self.layer_output = layer_output
        self.W_out = nn.ModuleList([nn.Linear(2 * dim, dim), nn.Linear(dim, 128)])  # Output layers
        self.W_interaction = nn.Linear(128, 2)  # Final interaction layer

    def forward(self, batch, datasetF):
        # Forward pass through the model

        # Get drug and protein vectors through MGCN layer
        x_d_dr, y_p_pro = self.gcnlayer(datasetF)

        # Extract batch indices and interactions
        id0 = batch[:, 0].type(torch.long).to(device)
        id1 = batch[:, 1].type(torch.long).to(device)
        interaction = batch[:, 2].type(torch.long).to(device)

        # Extract drug and protein features from the graph
        drugs = x_d_dr[id0, :, :]
        proteins = y_p_pro[id1, :, :]

        # Apply each layer in DIA_ModuleList for interaction learning
        for i in range(self.layer_IA):
            drug_vector, protein_vector = self.DIA_ModuleList[i](drugs, proteins)

            if i == 0:
                drug_vector_co, protein_vector_co = torch.cat([drugs, drug_vector], dim=-1), torch.cat(
                    [proteins, protein_vector], dim=-1)
            else:
                drug_vector_co, protein_vector_co = torch.cat([drug_vector_co, drug_vector], dim=-1), torch.cat(
                    [protein_vector_co, protein_vector], dim=-1)

        # Pass the concatenated vectors through the linear layers
        drug_vector, protein_vector = self.dr_lin(drug_vector_co), self.pro_lin(protein_vector_co)

        # Compute the drug and protein co-vectors (mean pooling)
        drug_covector = drug_vector.mean(dim=1)
        protein_covector = protein_vector.mean(dim=1)

        # Concatenate drug and protein co-vectors for interaction prediction
        cat_vector = torch.cat((drug_covector, protein_covector), 1)

        # Apply output layers and get the final prediction
        for j in range(self.layer_output - 1):
            cat_vector = torch.tanh(self.W_out[j](cat_vector))
        predicted = self.W_interaction(cat_vector)

        return drug_covector, protein_covector, predicted, interaction

    def save_fold(self, fold, train_drug_covector, train_protein_covector, train_labels, dev_drug_covector,
                  dev_protein_covector, dev_labels, predicted_train):
        # Save the results of each fold in the training process
        directory = './mid_data/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Convert tensors to numpy arrays for saving
        train_labels = train_labels.detach().cpu().numpy() if isinstance(train_labels, torch.Tensor) else np.array(
            train_labels)
        dev_labels = dev_labels.detach().cpu().numpy() if isinstance(dev_labels, torch.Tensor) else np.array(dev_labels)

        # Save the data using joblib
        joblib.dump({
            'train_drug_covector': train_drug_covector.detach().cpu().numpy(),
            'train_protein_covector': train_protein_covector.detach().cpu().numpy(),
            'train_labels': train_labels,
            'dev_drug_covector': dev_drug_covector.detach().cpu().numpy(),
            'dev_protein_covector': dev_protein_covector.detach().cpu().numpy(),
            'dev_labels': dev_labels,
            'predicted_train': predicted_train.detach().cpu().numpy(),
        }, os.path.join(directory, f'{fold}-sigmoid.npy'))

        print(f"Data for fold {fold} saved successfully.")
