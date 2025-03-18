from typing import Optional
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
from torch.nn import Linear, Parameter
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax  # Import softmax for proper indexing

class SGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.3, bias=True, **kwargs):
        super(SGATConv, self).__init__(node_dim=0, aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.lin(x)
        x = x.view(-1, self.heads, self.out_channels)
        x = F.dropout(x, p=self.dropout, training=self.training)  # NEW: Apply dropout before attention

        x = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        if self.bias is not None:
            x = x + self.bias
        return F.elu(x)  # Activation at the end

    def message(self, x_i, x_j, index, edge_weight=None):
        x = torch.cat([x_i, x_j], dim=-1)
        alpha = (x * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        alpha = softmax(alpha, index, num_nodes=x_i.shape[0])  # Normalize per source node
        return alpha.unsqueeze(-1) * x_j  # Weighted features

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce="mean")  # CHANGED: Mean instead of sum

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})"
