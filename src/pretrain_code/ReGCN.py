import math
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.pooling import AvgPool2d
from torch.nn.modules.conv import Conv2d
from torch.nn.parameter import Parameter

# Set device to CUDA if available, otherwise default to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the Graph Convolution layer
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # Weight for the GCN layer
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))  # Bias term, optional
        else:
            self.register_parameter('bias', None)  # Register the bias as None if not used
        self.reset_parameters()  # Initialize parameters

    def reset_parameters(self):
        # Initialize weights with a small random value
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, input):
        # Perform graph convolution: A * X * W
        input = input.float()  # Ensure input is float
        support = torch.matmul(input, self.weight)  # Linear transformation
        output = torch.matmul(adj, support)  # Apply adjacency matrix
        if self.bias is not None:
            return output + self.bias  # Add bias if present
        else:
            return output


# Define the ReGCNs model for both drug and protein data
class ReGCNs(nn.Module):
    def __init__(self, args):
        super(ReGCNs, self).__init__()
        self.args = args  # Store the arguments

        # GCN layers for drug-related features
        self.gcn_x1_dr = GraphConvolution(args.drug_number, args.f)
        self.gcn_x2_dr = GraphConvolution(args.f, args.f)
        self.gcn_x1_di = GraphConvolution(args.drug_number, args.f)
        self.gcn_x2_di = GraphConvolution(args.f, args.f)
        self.gcn_x1_se = GraphConvolution(args.drug_number, args.f)
        self.gcn_x2_se = GraphConvolution(args.f, args.f)
        self.gcn_x1_str = GraphConvolution(args.drug_number, args.f)
        self.gcn_x2_str = GraphConvolution(args.f, args.f)
        self.gcn_x1_pro = GraphConvolution(args.drug_number, args.f)
        self.gcn_x2_pro = GraphConvolution(args.f, args.f)
        self.gcn_x1_gs = GraphConvolution(args.drug_number, args.f)
        self.gcn_x2_gs = GraphConvolution(args.f, args.f)

        # GCN layers for protein-related features
        self.gcn_y1_pro = GraphConvolution(args.protein_number, args.f)
        self.gcn_y2_pro = GraphConvolution(args.f, args.f)
        self.gcn_y1_di = GraphConvolution(args.protein_number, args.f)
        self.gcn_y2_di = GraphConvolution(args.f, args.f)
        self.gcn_y1_seq = GraphConvolution(args.protein_number, args.f)
        self.gcn_y2_seq = GraphConvolution(args.f, args.f)
        self.gcn_y1_dr = GraphConvolution(args.protein_number, args.f)
        self.gcn_y2_dr = GraphConvolution(args.f, args.f)
        self.gcn_y1_gs = GraphConvolution(args.protein_number, args.f)
        self.gcn_y2_gs = GraphConvolution(args.f, args.f)

        # Pooling layers for aggregating features
        self.globalAvgPool_x = AvgPool2d((args.f, args.drug_number), (1, 1))
        self.globalAvgPool_y = AvgPool2d((args.f, args.protein_number), (1, 1))

        # Fully connected layers for feature transformation
        self.fc1_x = nn.Linear(6, 5 * 6)
        self.fc2_x = nn.Linear(5 * 6, 6)
        self.fc1_y = nn.Linear(5, 25)
        self.fc2_y = nn.Linear(25, 5)

        # Final output layers for drug and protein predictions
        self.f_d = nn.Linear(args.f, args.d_out_channels)
        self.f_p = nn.Linear(args.f, args.p_out_channels)

        # Activation functions
        self.sigmoidx = Sigmoid()
        self.sigmoidy = Sigmoid()

        # Convolutional layers for drug and protein feature maps
        self.cnn_drug = Conv2d(6, args.d_out_channels, kernel_size=(args.f, 1), stride=1, bias=True)
        self.cnn_pro = Conv2d(5, args.p_out_channels, kernel_size=(args.f, 1), stride=1, bias=True)

    def forward(self, data):
        torch.manual_seed(1)  # Fixing seed for reproducibility

        if self.args.name == 'drug':
            x_d = torch.eye(self.args.drug_number).to(device)  # Identity matrix for drug data

            # Pass drug data through GCN layers
            x_d_dr2 = torch.relu(self.gcn_x2_dr(data['dd_dr']['data_matrix'].to(device), torch.relu(self.gcn_x1_dr(data['dd_dr']['data_matrix'].to(device), x_d))))
            x_d_di2 = torch.relu(self.gcn_x2_di(data['dd_di']['data_matrix'].to(device), torch.relu(self.gcn_x1_di(data['dd_di']['data_matrix'].to(device), x_d))))
            x_d_se2 = torch.relu(self.gcn_x2_se(data['dd_se']['data_matrix'].to(device), torch.relu(self.gcn_x1_se(data['dd_se']['data_matrix'].to(device), x_d))))
            x_d_str2 = torch.relu(self.gcn_x2_str(data['dd_str']['data_matrix'].to(device), torch.relu(self.gcn_x1_str(data['dd_str']['data_matrix'].to(device), x_d))))
            x_d_pro2 = torch.relu(self.gcn_x2_pro(data['dd_pro']['data_matrix'].to(device), torch.relu(self.gcn_x1_pro(data['dd_pro']['data_matrix'].to(device), x_d))))
            x_d_gs2 = torch.relu(self.gcn_x2_gs(data['dd_gs']['data_matrix'].to(device), torch.relu(self.gcn_x1_gs(data['dd_gs']['data_matrix'].to(device), x_d))))

            # Concatenate the output features
            XM = torch.cat((x_d_dr2, x_d_di2, x_d_se2, x_d_str2, x_d_pro2, x_d_gs2), dim=1).t()
            XM = XM.contiguous().reshape(1, 6, self.args.f, -1)

            # Apply global average pooling and attention mechanism
            x_att = self.globalAvgPool_x(XM).view(1, -1)
            x_att = self.sigmoidx(self.fc2_x(torch.relu(self.fc1_x(x_att))))
            x_att = x_att.view(1, 6, 1, 1)
            XM = XM * x_att.expand_as(XM)

            # Apply convolution for drug feature extraction
            x_d_fe = self.cnn_drug(XM).view(self.args.d_out_channels, self.args.drug_number).t()
            x_d = x_d_fe @ x_d_fe.t()  # Compute the similarity matrix
            return x_d_fe, x_d  # Return feature map and similarity matrix

        elif self.args.name == 'protein':
            y_p = torch.eye(self.args.protein_number).to(device)  # Identity matrix for protein data

            # Pass protein data through GCN layers
            y_p_pro2 = torch.relu(self.gcn_y2_pro(data['pp_pro']['data_matrix'].to(device), torch.relu(self.gcn_y1_pro(data['pp_pro']['data_matrix'].to(device), y_p))))
            y_p_di2 = torch.relu(self.gcn_y2_di(data['pp_di']['data_matrix'].to(device), torch.relu(self.gcn_y1_di(data['pp_di']['data_matrix'].to(device), y_p))))
            y_p_seq2 = torch.relu(self.gcn_y2_seq(data['pp_seq']['data_matrix'].to(device), torch.relu(self.gcn_y1_seq(data['pp_seq']['data_matrix'].to(device), y_p))))
            y_p_dr2 = torch.relu(self.gcn_y2_dr(data['pp_dr']['data_matrix'].to(device), torch.relu(self.gcn_y1_dr(data['pp_dr']['data_matrix'].to(device), y_p))))
            y_p_gs2 = torch.relu(self.gcn_y2_gs(data['pp_gs']['data_matrix'].to(device), torch.relu(self.gcn_y1_gs(data['pp_gs']['data_matrix'].to(device), y_p))))

            # Concatenate the output features
            YM = torch.cat((y_p_pro2, y_p_di2, y_p_seq2, y_p_dr2, y_p_gs2), dim=1).t()
            YM = YM.contiguous().reshape(1, 5, self.args.f, -1)

            # Apply global average pooling and attention mechanism
            y_att = self.globalAvgPool_y(YM).view(1, -1)
            y_att = self.sigmoidy(self.fc2_y(torch.relu(self.fc1_y(y_att))))
            y_att = y_att.view(1, 5, 1, 1)
            YM = YM * y_att.expand_as(YM)

            # Apply convolution for protein feature extraction
            y_p_fe = self.cnn_pro(YM).view(self.args.p_out_channels, self.args.protein_number).t()
            y_p = y_p_fe @ y_p_fe.t()  # Compute the similarity matrix
            return y_p_fe, y_p  # Return feature map and similarity matrix
