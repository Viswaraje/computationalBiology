from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch.nn.init import xavier_uniform_, zeros_


# Custom Initializers
def glorot(tensor):
    if tensor is not None:
        xavier_uniform_(tensor)

def zeros(tensor):
    if tensor is not None:
        zeros_(tensor)

class GATConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4, concat: bool = True, dropout: float = 0.6,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        super(GATConv, self).__init__(aggr='add', **kwargs)  # Using 'add' aggregation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.add_self_loops = add_self_loops
        self.dropout = dropout

        # Weight matrix for node feature transformation
        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
        # Learnable attention parameters
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(heads * out_channels if concat else out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters"""
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        """Forward pass through the GAT layer"""
        if self.add_self_loops:
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=x.size(0))

        # Linear transformation
        x = torch.matmul(x, self.weight).view(-1, self.heads, self.out_channels)

        # Perform message passing
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        if self.bias is not None:
            out += self.bias  # Apply bias

        return F.elu(out)  # Apply ELU activation

    def message(self, x_i: Tensor, x_j: Tensor, index: Tensor, size: Optional[int] = None, edge_weight: OptTensor = None) -> Tensor:
        """Compute attention scores and update node features"""
        # Compute attention scores
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)

        # Normalize attention scores per source node
        alpha = softmax(alpha, index, num_nodes=x_i.shape[0])  # Normalize first

        # Apply edge weights if available (after softmax)
        if edge_weight is not None:
            alpha = alpha * edge_weight.view(-1, 1)

        # Apply dropout to attention coefficients
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Weight node features with attention scores
        return x_j * alpha.unsqueeze(-1)

    def update(self, aggr_out: Tensor) -> Tensor:
        """Update the node embeddings after aggregation"""
        if self.concat:
            return F.elu(aggr_out.view(-1, self.heads * self.out_channels))  # Apply ELU
        else:
            return F.elu(aggr_out.mean(dim=1))  # Average attention heads

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)
