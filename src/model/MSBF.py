import torch.nn as nn
import torch.nn.functional as F
import torch
import math


# --- Multi-Scale Bilateral Fusion (MSBF) Layer ---
class MSBF(nn.Module):
    """
    MSBF (Multi-Scale Bilateral Fusion) Layer for feature extraction and aggregation from multiple scales.
    It uses Multi-Head Attention (MHAtt), dynamic gating, and multiple convolution operations for multi-scale feature fusion.
    """

    def __init__(self, hid_dim, n_heads, dropout):
        super(MSBF, self).__init__()

        # Multi-Head Attention Layer
        self.mhatt = MHAtt(hid_dim, n_heads, dropout)

        # Dropout and Layer Normalization for regularization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hid_dim)

        # Dynamic Gate to control the fusion of features
        self.gate = DynamicGate(hid_dim * 2)  # Bilateral gating mechanism

        # Additional convolution layers for multi-scale aggregation
        self.scale1 = nn.Conv1d(hid_dim, hid_dim, kernel_size=1)
        self.scale2 = nn.Conv1d(hid_dim, hid_dim, kernel_size=3, padding=1)
        self.scale3 = nn.Conv1d(hid_dim, hid_dim, kernel_size=5, padding=2)

        # Fully connected layer to combine multi-scale features
        self.fc = nn.Linear(hid_dim * 3, hid_dim)

    def forward(self, x, y, y_mask=None):
        """
        Forward pass through the MSBF layer, which applies multi-head attention,
        followed by multi-scale convolution, and dynamic gating for feature fusion.
        """

        # Apply attention and normalization
        x = self.norm(x + self.dropout(self.mhatt(y, y, x, y_mask)))

        # Multi-scale feature extraction using convolutions
        x1 = self.scale1(x.transpose(1, 2)).transpose(1, 2)
        x2 = self.scale2(x.transpose(1, 2)).transpose(1, 2)
        x3 = self.scale3(x.transpose(1, 2)).transpose(1, 2)

        # Dynamic gate to combine multi-scale features
        gate_weight = self.gate(x1, x2)  # First gate fusion between x1 and x2
        output = self.gate(gate_weight, x3)  # Second gate fusion with x3

        # Apply ReLU activation after gate fusion
        output = F.relu(output)

        # Normalize the output before returning
        x = self.norm(output)
        return x


# --- Dynamic Gate for Feature Fusion ---
class DynamicGate(nn.Module):
    """
    Dynamic Gate module to control the fusion of two sets of features based on learned weights.
    """

    def __init__(self, input_dim):
        super().__init__()
        # Sequential layers to compute gate weights
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.SiLU(),  # SiLU activation for smoother learning
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()  # Sigmoid activation to output values between 0 and 1
        )

    def forward(self, *inputs):
        """
        Forward pass through the gate, where inputs are fused using dynamic gate weights.
        """
        combined = torch.cat(inputs, dim=-1)  # Concatenate inputs along the last dimension
        gate_weights = self.gate_net(combined)  # Compute gate weights for fusion
        return gate_weights * inputs[0] + (1 - gate_weights) * inputs[1]  # Perform weighted fusion


# --- Multi-Head Attention Layer ---
class MHAtt(nn.Module):
    """
    Multi-Head Attention Layer used in MSBF for attention-based feature aggregation.
    This layer uses self-attention to compute the importance of features across different heads.
    """

    def __init__(self, hid_dim, n_heads, dropout):
        super(MHAtt, self).__init__()

        # Linear transformations for value (v), key (k), and query (q)
        self.linear_v = nn.Linear(hid_dim, hid_dim)
        self.linear_k = nn.Linear(hid_dim, hid_dim)
        self.linear_q = nn.Linear(hid_dim, hid_dim)

        # Final linear transformation for the aggregated attention output
        self.linear_merge = nn.Linear(hid_dim, hid_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        self.hid_dim = hid_dim
        self.nhead = n_heads

        # Compute hidden size per head
        self.hidden_size_head = int(self.hid_dim / self.nhead)

    def forward(self, v, k, q, mask):
        """
        Forward pass through the Multi-Head Attention mechanism.
        Computes attention scores and applies them to the value (v).
        """

        n_batches = q.size(0)

        # Apply linear transformations and reshape for multi-head attention
        v = self.linear_v(v).view(n_batches, -1, self.nhead, self.hidden_size_head).transpose(1, 2)
        k = self.linear_k(k).view(n_batches, -1, self.nhead, self.hidden_size_head).transpose(1, 2)
        q = self.linear_q(q).view(n_batches, -1, self.nhead, self.hidden_size_head).transpose(1, 2)

        # Compute the attention scores and apply them to the value
        atted = self.att(v, k, q, mask)

        # Reshape the attention output and merge across all heads
        atted = atted.transpose(1, 2).contiguous().view(n_batches, -1, self.hid_dim)
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask=None):
        """
        Attention mechanism to compute attention scores and apply them to the value tensor.
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Scaled dot-product attention

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)  # Apply mask if provided

        # Compute attention map using softmax
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)  # Apply dropout to attention map

        # Multiply attention map with value to get the weighted sum
        return torch.matmul(att_map, value)


