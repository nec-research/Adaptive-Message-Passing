import torch
from pydgn.model.interface import ModelInterface

from torch_geometric.nn import global_add_pool, global_max_pool, \
    global_mean_pool
from torch.nn import Module, Linear, Sequential, LeakyReLU
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.data import Data
from collections import OrderedDict
from torch import Tensor, tanh
from typing import Optional, Callable


class DGCConv(MessagePassing):
    def __init__(self,
                 input_dim: int,
                 epsilon: float = 0.1,
                 iterations: int = 1,
                 cached: bool = False,
                 add_self_loops: bool = True) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.epsilon = epsilon
        self.iterations = iterations
        self.cached = cached
        self.add_self_loops = add_self_loops

        self.cached_edge_index = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if self.cached_edge_index is None:
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight=edge_weight,
                num_nodes=x.size(self.node_dim), improved=True,
                add_self_loops=self.add_self_loops, dtype=x.dtype)

            if self.cached:
                self.cached_edge_index = (edge_index, edge_weight)
        else:
            edge_index, edge_weight = self.cached_edge_index

        for _ in range(self.iterations):
            epsilon_LX = self.epsilon * self.propagate(edge_index, x=x,
                                                       edge_weight=edge_weight)
            x = x - epsilon_LX

        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


class DGC_GraphProp(ModelInterface):
    def __init__(
        self,
        dim_node_features: int,
        dim_edge_features: int,
        dim_target: int,
        readout_class: Callable[..., torch.nn.Module],
        config: dict,
    ):
        super().__init__(
            dim_node_features,
            dim_edge_features,
            dim_target,
            readout_class,
            config,
        )

        self.input_dim = dim_node_features
        self.output_dim = dim_target
        self.hidden_dim = config['hidden_dim']
        self.epsilon = config['epsilon']
        self.iterations = config['iterations']
        self.cached = config['cached']
        self.add_self_loops = True #config['add_self_loops']
        self.global_aggregation = config['global_aggregation']

        inp = self.input_dim
        self.emb = None
        if self.hidden_dim is not None:
            self.emb = Linear(self.input_dim, self.hidden_dim)
            inp = self.hidden_dim

        self.conv = DGCConv(inp, self.epsilon, self.iterations, self.cached,
                            self.add_self_loops)

        self.readout = Linear(inp, self.output_dim)

        if not self.global_aggregation:
            self.readout = Sequential(OrderedDict([
                ('L1', Linear(inp, inp // 2)),
                ('LeakyReLU1', LeakyReLU()),
                ('L2', Linear(inp // 2, self.output_dim)),
                ('LeakyReLU2', LeakyReLU())
            ]))
        else:
            self.readout = Sequential(OrderedDict([
                ('L1', Linear(inp * 3, (inp * 3) // 2)),
                ('LeakyReLU1', LeakyReLU()),
                ('L2', Linear((inp * 3) // 2, self.output_dim)),
                ('LeakyReLU2', LeakyReLU())
            ]))

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h_list = []

        x = self.emb(x) if self.emb else x

        x = tanh(self.conv(x, edge_index))

        h_list.append(x)

        if self.global_aggregation:
            x = torch.cat(
                [global_add_pool(x, batch), global_max_pool(x, batch),
                 global_mean_pool(x, batch)], dim=1)
        x = self.readout(x)

        h_list = torch.stack(h_list, dim=1)

        return x, h_list, [batch]