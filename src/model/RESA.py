import torch.nn as nn
import torch.nn.functional as F
import torch
import math

# --- RESA Module ---
class RESA(nn.Module):

    def __init__(self, hid_dim, n_heads, dropout):
        super(RESA, self).__init__()

        # Initialize multi-head attention and normalization layers
        self.mhatt = MSHAtt(hid_dim, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, x, mask=None):
        """
        Forward pass through the RESA module.
        x: input feature tensor.
        mask: optional attention mask.
        """
        residual = x  # Store the input for residual connection

        # Apply multi-head attention and normalization with dropout
        x = self.norm(x + self.dropout(self.mhatt(x, x, x, mask)))

        # Add residual connection and return the output
        return x + residual


# --- MSHAtt (Multi-Scale Head Attention) Module ---
class MSHAtt(nn.Module):
    """
    Multi-Scale Head Attention (MSHAtt) performs multi-head self-attention with dynamic gating and multi-scale convolution.
    """

    def __init__(self, hid_dim, n_heads, dropout):
        super(MSHAtt, self).__init__()

        # Initialize linear transformations for value, key, and query in multi-head attention
        self.linear_v = nn.Linear(hid_dim, hid_dim)
        self.linear_k = nn.Linear(hid_dim, hid_dim)
        self.linear_q = nn.Linear(hid_dim, hid_dim)
        self.linear_merge = nn.Linear(hid_dim, hid_dim)

        # Parameters for multi-head attention
        self.hid_dim = hid_dim
        self.nhead = n_heads
        self.hidden_size_head = int(self.hid_dim / self.nhead)
        self.dropout = nn.Dropout(dropout)

        # MLP to generate dynamic alpha for feature fusion
        self.alpha_mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.ReLU(),
            nn.Linear(hid_dim // 2, 1),
            nn.Sigmoid()  # Output alpha in the range [0, 1]
        )

        # Multi-scale convolution layers for feature extraction
        self.conv1 = nn.Conv1d(hid_dim, hid_dim, kernel_size=1)
        self.conv3 = nn.Conv1d(hid_dim, hid_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(hid_dim, hid_dim, kernel_size=5, padding=2)

        # 1x1 convolution to merge multi-scale features
        self.fc = nn.Conv1d(hid_dim * 3, hid_dim, kernel_size=1)

    def forward(self, v, k, q, mask):
        """
        Forward pass through multi-head attention with dynamic gating and multi-scale convolution.
        v, k, q: value, key, query tensors.
        mask: optional attention mask.
        """
        n_batches = q.size(0)

        v_orig = v  # Save original value for residual connection

        # Apply linear transformations for value, key, and query, and reshape for multi-head attention
        v = self.linear_v(v).view(n_batches, -1, self.nhead, self.hidden_size_head).transpose(1, 2)
        k = self.linear_k(k).view(n_batches, -1, self.nhead, self.hidden_size_head).transpose(1, 2)
        q = self.linear_q(q).view(n_batches, -1, self.nhead, self.hidden_size_head).transpose(1, 2)

        # Compute attention using scaled dot-product
        atted = self.att(v, k, q, mask)

        # Reshape and merge the attention output from all heads
        atted = atted.transpose(1, 2).contiguous().reshape(n_batches, -1, self.hid_dim)

        # Apply multi-scale convolutions
        multi_scale_1 = F.relu(self.conv1(atted.transpose(1, 2)))
        multi_scale_3 = F.relu(self.conv3(atted.transpose(1, 2)))
        multi_scale_5 = F.relu(self.conv5(atted.transpose(1, 2)))

        # Concatenate multi-scale features
        multi_scale_concat = torch.cat([multi_scale_3, multi_scale_5, multi_scale_1], dim=1)

        # Fuse the multi-scale features using 1x1 convolution
        multi_scale_output = F.relu(self.fc(multi_scale_concat))

        # Restore the original dimensions
        multi_scale_output = multi_scale_output.transpose(1, 2)

        # Compute dynamic alpha based on feature means
        alpha = self.alpha_mlp(atted.mean(dim=1))
        alpha = alpha.view(-1, 1, 1)  # Reshape alpha for broadcasting

        # Fuse the multi-scale output and original attention using alpha
        atted = alpha * multi_scale_output + (1 - alpha) * atted

        # Final linear transformation to merge attention features
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask=None):
        """
        Scaled dot-product attention mechanism.
        value, key, query: Input tensors.
        mask: Optional attention mask.
        """
        d_k = query.size(-1)

        # Compute attention scores (scaled dot product)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        # Compute the attention map using softmax
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        # Apply attention map to value to get the final attention output
        return torch.matmul(att_map, value)


