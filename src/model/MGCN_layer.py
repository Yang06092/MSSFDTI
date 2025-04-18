import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import Module

# --- NGNN (Multi-hop Graph Neural Network) Layer ---
class NGNN(Module):
    """
    NGNN (Multi-hop Graph Neural Network) Layer for graph convolution.
    This layer performs graph convolution for a given order (multi-hop).
    """

    def __init__(self, in_features, out_features, order=1, bias=True):
        super(NGNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.order = order
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.act = nn.Tanh()  # Activation function (can replace with elu or prelu if needed)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize weights and bias
        torch.nn.init.xavier_uniform_(self.weight)
        stdv = 1. / math.sqrt(self.weight.size(1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, features, adj, active=True):
        # Perform graph convolution by applying weight matrix and activation function
        output = torch.mm(features, self.weight)
        if active:
            output = self.act(output)
        output = torch.spmm(adj, output)  # Apply adjacency matrix (graph convolution)

        # Perform higher-order graph convolution (multi-hop)
        for _ in range(self.order - 1):
            output = torch.spmm(adj, output)

        # Add bias if present
        if self.bias is not None:
            return output + self.bias
        else:
            return output


# --- Graph Convolution Layer ---
class GraphConvolution(Module):
    """
    Graph Convolution Layer using NGNN.
    This layer applies multiple graph convolution operations (order) and combines the results.
    """

    def __init__(self, in_features, out_features, reduction=4, order=3):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.order = order

        # Create multiple NGNN layers for different orders
        self.main_layer = [NGNN(self.in_features, self.out_features, i) for i in range(1, self.order + 1)]
        self.main_layers = torch.nn.ModuleList(self.main_layer)

        # Fully connected layers to reduce the features
        self.fc1 = nn.Linear(out_features, out_features // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_features // reduction, out_features)

        # Softmax for the final output
        self.softmax = nn.Softmax(dim=0)

    def forward(self, adj, features, active=True):
        # Process features through different graph convolutions (multi-hop)
        abstract_features = [self.main_layers[i](features, adj, active=active) for i in range(self.order)]

        # Compute mean feature vectors for each order of convolution
        feats_mean = [torch.mean(abstract_features[i], 0, keepdim=True) for i in range(self.order)]

        # Concatenate the mean features
        feats_mean = torch.cat(feats_mean, dim=0)

        # Apply fully connected layers to compute the importance of each feature
        feats_a = self.fc2(self.relu(self.fc1(feats_mean)))
        feats_a = self.softmax(feats_a)

        # Weight each feature layer by its importance
        feats = []
        for i in range(self.order):
            feats.append(abstract_features[i] * feats_a[i])

        # Sum all weighted features to get the final output
        output = feats[0]
        for i in range(1, self.order):
            output += feats[i]

        return output


# --- Heterogeneous Graph Encoder ---
class GCN_hete(nn.Module):
    """
    GCN for heterogeneous graph encoder.
    This encoder works with multiple types of relationships between nodes.
    """

    def __init__(self, hidden_c, output_hete):
        super(GCN_hete, self).__init__()

        self.hete = GraphConvolution(hidden_c, output_hete)
        self.hete2 = GraphConvolution(output_hete, output_hete)
        self.hete3 = GraphConvolution(output_hete, output_hete)

    def forward(self, adj_c, features_c):
        # Apply graph convolutions for heterogeneous graph
        out1 = torch.relu(self.hete(adj_c, features_c))
        out2 = torch.relu(self.hete2(adj_c, out1))
        out3 = torch.relu(self.hete3(adj_c, out2))
        return out1, out2, out3


# --- Homogeneous Graph Encoder ---
class GCN_homo(nn.Module):
    """
    GCN for homogeneous graph encoder.
    This encoder works with a single type of relationship between nodes (e.g., drug-drug, protein-protein).
    """

    def __init__(self, hidden_homo, output_homo):
        super(GCN_homo, self).__init__()

        self.gcn_homo = GraphConvolution(hidden_homo, output_homo)
        self.gcn_homo2 = GraphConvolution(output_homo, output_homo)
        self.gcn_homo3 = GraphConvolution(output_homo, output_homo)

    def forward(self, adj, features):
        # Apply graph convolutions for homogeneous graph
        out_d1 = torch.relu(self.gcn_homo(adj, features))
        out_d2 = torch.relu(self.gcn_homo2(adj, out_d1))
        out_d3 = torch.relu(self.gcn_homo3(adj, out_d2))
        return out_d1, out_d2, out_d3


# --- Bipartite Graph Encoder ---
class GCN_bi(nn.Module):
    """
    GCN for bipartite graph encoder.
    This encoder works with a bipartite graph (e.g., drug-protein interactions).
    """

    def __init__(self, hidden_bi, output_bi):
        super(GCN_bi, self).__init__()

        # Bipartite graph: drug-protein interactions
        self.gcn_bi_dp = GraphConvolution(hidden_bi, output_bi)
        self.gcn_bi_dp2 = GraphConvolution(output_bi, output_bi)
        self.gcn_bi_dp3 = GraphConvolution(output_bi, output_bi)

    def forward(self, adj_dp, features_dp):
        # Apply graph convolutions for bipartite graph
        out_dp1 = torch.relu(self.gcn_bi_dp(adj_dp, features_dp))
        out_dp2 = torch.relu(self.gcn_bi_dp2(adj_dp, out_dp1))
        out_dp3 = torch.relu(self.gcn_bi_dp3(adj_dp, out_dp2))
        return out_dp1, out_dp2, out_dp3


