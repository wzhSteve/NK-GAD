import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN, SAGEConv, PNAConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_scipy_sparse_matrix, to_dense_adj
import matplotlib.pyplot as plt
import numpy as np
import scipy

from torch_geometric.nn.models.basic_gnn import BasicGNN
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError
        
    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias

def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity


# Define GAT-based aggregation for both mean and covariance
class GATAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super(GATAggregator, self).__init__()
        # self.gat_mean = GCNConv(in_channels=input_dim, out_channels=input_dim)
        # self.gat_cov = GCNConv(in_channels=input_dim * input_dim, out_channels=input_dim * input_dim)
        self.gat_mean = GATConv(input_dim, hidden_dim, heads=heads, concat=False)  # For mean
        self.gat_cov = GATConv(input_dim * input_dim, hidden_dim * hidden_dim, heads=heads, concat=False)  # For covariance
        # self.fc_mean = nn.Linear(hidden_dim, output_dim)
        # self.fc_cov = nn.Linear(hidden_dim * hidden_dim, hidden_dim * hidden_dim)
        self.fc_mean = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Linear(hidden_dim, output_dim))
        self.fc_cov = nn.Sequential(nn.Linear(input_dim * input_dim, hidden_dim * hidden_dim), nn.Linear(hidden_dim * hidden_dim, output_dim * output_dim))

    def forward(self, mean, cov, edge_index):
        cov_flat = cov.view(cov.size(0), -1)
        mean_agg = self.gat_mean(mean, edge_index)
        cov_agg = self.gat_cov(cov_flat, edge_index)
        mean_out = self.fc_mean(mean_agg)
        cov_out = self.fc_cov(cov_agg)
        cov_out = cov_out.view(cov.size(0), cov.size(1), cov.size(2))
        return mean_out, cov_out


class HighPassConv(MessagePassing):
    def __init__(self, in_channels, out_channels, improved=False, cached=False, add_self_loops=True, bias=True, **kwargs):
        super(HighPassConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        if self.add_self_loops:
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=x.size(0))
        
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=x.dtype, device=edge_index.device)
        
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        norm = -norm  # Flip the sign for (I - A)
        identity_index = torch.arange(0, x.size(0), device=x.device)
        identity_weight = torch.ones(x.size(0), device=x.device)

        edge_index = torch.cat([edge_index, torch.stack([identity_index, identity_index], dim=0)], dim=1)
        edge_weight = torch.cat([norm, identity_weight])
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = torch.matmul(out, self.weight)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

class HighPassGCN(BasicGNN):
    r"""The Graph Neural Network from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper, using the
    :class:`~torch_geometric.nn.conv.GCNConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GCNConv`.
    """
    supports_edge_weight: Final[bool] = True
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return HighPassConv(in_channels, out_channels, **kwargs)


class LowPassConv(MessagePassing):
    def __init__(self, in_channels, out_channels, improved=False, cached=False, add_self_loops=True, bias=True, **kwargs):
        super(LowPassConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        if self.add_self_loops:
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=x.size(0))
        
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=x.dtype, device=edge_index.device)

        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        identity_index = torch.arange(0, x.size(0), device=x.device)
        identity_weight = torch.ones(x.size(0), device=x.device)

        edge_index = torch.cat([edge_index, torch.stack([identity_index, identity_index], dim=0)], dim=1)
        edge_weight = torch.cat([norm, identity_weight])

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = torch.matmul(out, self.weight)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)



class LowPassGCN(BasicGNN):
    r"""The Graph Neural Network from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper, using the
    :class:`~torch_geometric.nn.conv.GCNConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GCNConv`.
    """
    supports_edge_weight: Final[bool] = True
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return LowPassConv(in_channels, out_channels, **kwargs)


def compute_spectrum(adj_matrix):
    laplacian = scipy.sparse.csgraph.laplacian(adj_matrix, normed=True)
    eigenvalues, _ = np.linalg.eigh(laplacian.toarray())
    return eigenvalues

def plot_spectrum(title, eigenvalues_before, eigenvalues_after):
    plt.figure(figsize=(10, 6))
    plt.plot(eigenvalues_before, label="Original Signal", marker='o')
    plt.plot(eigenvalues_after, label="Filtered Signal", marker='x')
    plt.title(title)
    plt.xlabel("Frequency Component Index")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)
    plt.savefig(title + ".pdf")
