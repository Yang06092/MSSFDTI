import torch
import numpy as np
import scipy.sparse as sp
# Import read_csv function to read CSV files
from .torch_data import read_csv

# Preprocess adjacency matrix by adding self-loops and normalizing it
def preprocess_adj(adj):
    """Preprocess the adjacency matrix (for a simple GCN model) and convert to tuple representation."""
    adj_normalized = normalize_adj(adj + np.eye(adj.shape[0]))  # Add self-loops
    return torch.Tensor(adj_normalized)  # Return as Tensor

# New version of adjacency preprocessing without self-loops
def preprocess_adj_new(adj):
    """Normalize the adjacency matrix."""
    adj_normalized = normalize_adj(adj)
    return torch.Tensor(adj_normalized)  # Return normalized adjacency matrix

# Symmetric normalization of the adjacency matrix
def normalize_adj(adj):
    """Symmetrically normalize the adjacency matrix."""
    adj = sp.coo_matrix(adj)  # Convert to sparse COO format
    rowsum = np.array(adj.sum(1))  # Calculate degree for each node
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # Compute inverse square root of degrees
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # Avoid division by zero (set inf to 0)
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # Create inverse degree diagonal matrix
    # Normalize adjacency matrix by degree matrices
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).toarray()

# Normalize feature matrix based on node degrees
def normalize_features(feat):
    """Normalize feature matrix by node degrees."""
    degree = np.asarray(feat.sum(1)).flatten()  # Degree of each node
    degree[degree == 0.] = np.inf  # Avoid division by zero
    degree_inv = 1./ degree  # Inverse of the degree
    degree_inv_mat = sp.diags([degree_inv], [0])  # Degree inverse matrix
    feat_norm = degree_inv_mat.dot(feat)  # Normalize feature matrix
    return torch.Tensor(feat_norm)  # Return normalized features as Tensor

# Convert adjacency matrix to edge index for graph representation
def get_edge_index(matrix):
    edge_index = [[], []]  # Initialize lists for edge indices
    for i in range(matrix.size(0)):  # Loop over rows
        for j in range(matrix.size(1)):  # Loop over columns
            if matrix[i][j] != 0:  # If there's an edge
                edge_index[0].append(i)  # Add source node
                edge_index[1].append(j)  # Add target node
    return torch.LongTensor(edge_index)  # Return as LongTensor

# Prepare dataset based on the dataset type (drug-related or protein-related)
def data_pre(args):
    dataset = dict()  # Initialize empty dataset dictionary

    if args.name == 'drug':  # Process drug-related datasets
        "sim drug drug "
        dd_dr_matrix = read_csv(args.pretrain_dataset_path + 'Sim_mat_drug_drug.csv')  # Read similarity matrix
        dd_dr_matrix = preprocess_adj(dd_dr_matrix)  # Preprocess adjacency matrix
        dd_dr_edge_index = get_edge_index(dd_dr_matrix)  # Convert to edge index
        dataset['dd_dr'] = {'data_matrix': dd_dr_matrix, 'edges': dd_dr_edge_index}  # Store data

        "sim drug disease "
        dd_di_matrix = read_csv(args.pretrain_dataset_path + 'Sim_mat_drug_disease.csv')
        dd_di_matrix = preprocess_adj(dd_di_matrix)
        dd_di_edge_index = get_edge_index(dd_di_matrix)
        dataset['dd_di'] = {'data_matrix': dd_di_matrix, 'edges': dd_di_edge_index}

        "sim drug se"
        dd_se_matrix = read_csv(args.pretrain_dataset_path + 'Sim_mat_drug_se.csv')
        dd_se_matrix = preprocess_adj(dd_se_matrix)
        dd_se_edge_index = get_edge_index(dd_se_matrix)
        dataset['dd_se'] = {'data_matrix': dd_se_matrix, 'edges': dd_se_edge_index}

        "sim drug structure "
        dd_str_matrix = read_csv(args.pretrain_dataset_path + 'Sim_mat_drug_structure.csv')
        dd_str_matrix = preprocess_adj(dd_str_matrix)
        dd_str_edge_index = get_edge_index(dd_str_matrix)
        dataset['dd_str'] = {'data_matrix': dd_str_matrix, 'edges': dd_str_edge_index}

        "sim drug protein "
        dd_pro_matrix = read_csv(args.pretrain_dataset_path + 'Sim_mat_drug_protein.csv')
        dd_pro_matrix = preprocess_adj(dd_pro_matrix)
        dd_pro_edge_index = get_edge_index(dd_pro_matrix)
        dataset['dd_pro'] = {'data_matrix': dd_pro_matrix, 'edges': dd_pro_edge_index}

        "sim drug sig"
        dd_gs_matrix = read_csv(args.pretrain_dataset_path + 'drug_sigmoid.csv')  # Drug-protein similarity
        dd_gs_matrix = preprocess_adj(dd_gs_matrix)
        dd_gs_edge_index = get_edge_index(dd_gs_matrix)
        dataset['dd_gs'] = {'data_matrix': dd_gs_matrix, 'edges': dd_gs_edge_index}

    else:  # Process protein-related datasets
        "sim protein sig"
        pp_gs_matrix = read_csv(args.pretrain_dataset_path + 'target_sigmoid.csv')
        pp_gs_matrix = preprocess_adj(pp_gs_matrix)
        pp_gs_edge_index = get_edge_index(pp_gs_matrix)
        dataset['pp_gs'] = {'data_matrix': pp_gs_matrix, 'edges': pp_gs_edge_index}

        "sim protein protein"
        pp_pro_matrix = read_csv(args.pretrain_dataset_path + 'Sim_mat_protein_protein.csv')
        pp_pro_matrix = preprocess_adj(pp_pro_matrix)
        pp_pro_edge_index = get_edge_index(pp_pro_matrix)
        dataset['pp_pro'] = {'data_matrix': pp_pro_matrix, 'edges': pp_pro_edge_index}

        "sim protein disease"
        pp_di_matrix = read_csv(args.pretrain_dataset_path + 'Sim_mat_protein_disease.csv')
        pp_di_matrix = preprocess_adj(pp_di_matrix)
        pp_di_edge_index = get_edge_index(pp_di_matrix)
        dataset['pp_di'] = {'data_matrix': pp_di_matrix, 'edges': pp_di_edge_index}

        "sim protein sequence"
        pp_seq_matrix = read_csv(args.pretrain_dataset_path + 'Sim_mat_protein_sequence.csv')
        pp_seq_matrix = preprocess_adj(pp_seq_matrix)
        pp_seq_edge_index = get_edge_index(pp_seq_matrix)
        dataset['pp_seq'] = {'data_matrix': pp_seq_matrix, 'edges': pp_seq_edge_index}

        "sim protein drug"
        pp_dr_matrix = read_csv(args.pretrain_dataset_path + 'Sim_mat_protein_drug.csv')
        pp_dr_matrix = preprocess_adj(pp_dr_matrix)
        pp_dr_edge_index = get_edge_index(pp_dr_matrix)
        dataset['pp_dr'] = {'data_matrix': pp_dr_matrix, 'edges': pp_dr_edge_index}

    return dataset  # Return the prepared dataset
