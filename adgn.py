import math
from collections import OrderedDict
from typing import Optional, Callable
import torch
from pydgn.model.interface import ModelInterface
from torch.nn import Parameter, Linear, Sequential, LeakyReLU, ModuleList
from torch.nn.init import (
    kaiming_uniform_,
    _calculate_fan_in_and_fan_out,
    uniform_,
)
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_mean_pool, \
    global_max_pool

from torch_geometric.nn.conv import GCNConv, MessagePassing


class AntiSymmetricConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        num_iters: int = 1,
        gamma: float = 0.1,
        epsilon: float = 0.1,
        activ_fun: str = "tanh",  # it should be monotonically non-decreasing
        gcn_conv: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__(aggr="add")
        self.W = Parameter(
            torch.empty((in_channels, in_channels)), requires_grad=True
        )
        self.bias = (
            Parameter(torch.empty(in_channels), requires_grad=True)
            if bias
            else None
        )

        self.lin = Linear(
            in_channels, in_channels, bias=False
        )  # for simple aggregation
        self.I = Parameter(torch.eye(in_channels), requires_grad=False)

        self.gcn_conv = (
            GCNConv(in_channels, in_channels, bias=False) if gcn_conv else None
        )

        self.num_iters = num_iters
        self.gamma = gamma
        self.epsilon = epsilon
        self.activation = getattr(torch, activ_fun)

        self.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_filter: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        antisymmetric_W = self.W - self.W.T - self.gamma * self.I

        for _ in range(self.num_iters):
            if self.gcn_conv is None:
                # simple aggregation
                neigh_x = self.lin(x)
                neigh_x = self.propagate(
                    edge_index,
                    x=neigh_x,
                    edge_weight=edge_weight,
                    edge_filter=edge_filter,
                )
            else:
                # gcn aggregation
                neigh_x = self.gcn_conv(
                    x,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    edge_filter=edge_filter,
                )

            conv = x @ antisymmetric_W.T + neigh_x

            if self.bias is not None:
                conv += self.bias

            x = x + self.epsilon * self.activation(conv)
        return x

    def message(
        self,
        x_j: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_filter: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_weight is None and edge_filter is not None:
            return edge_filter * x_j

        elif edge_weight is None and edge_filter is None:
            return x_j

        elif edge_weight is not None and edge_filter is None:
            return edge_weight.view(-1, 1) * x_j

        else:
            return edge_filter * edge_weight.view(-1, 1) * x_j

    def reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        kaiming_uniform_(self.W, a=math.sqrt(5))
        self.lin.reset_parameters()
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            uniform_(self.bias, -bound, bound)


class GraphAntiSymmetricNN_GraphProp(ModelInterface):

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

        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.epsilon = config['adgn_epsilon']
        self.gamma = config['adgn_gamma']
        self.activ_fun = config['activ_fun']
        self.bias = config['bias']
        self.gcn_norm = config['gcn_norm']
        self.global_aggregation = config['global_aggregation']
        self.weight_sharing = config['weight_sharing']

        inp = self.dim_node_features
        self.emb = None
        if self.hidden_dim is not None:
            self.emb = Linear(self.dim_node_features, self.hidden_dim)
            inp = self.hidden_dim

        if self.weight_sharing:
            if self.dim_edge_features == 0:
                self.conv = AntiSymmetricConv(in_channels=inp,
                                              num_iters=self.num_layers,
                                              gamma=self.gamma,
                                              epsilon=self.epsilon,
                                              activ_fun=self.activ_fun,
                                              gcn_conv=self.gcn_norm,
                                              bias=self.bias)
            else:
                # DISCRETE EDGE TYPES ONLY
                self.conv = ModuleList()
                for _ in range(self.dim_edge_features):
                    self.conv.append = AntiSymmetricConv(in_channels=inp,
                                                  num_iters=self.num_layers,
                                                  gamma=self.gamma,
                                                  epsilon=self.epsilon,
                                                  activ_fun=self.activ_fun,
                                                  gcn_conv=self.gcn_norm,
                                                  bias=self.bias)
        else:
            if self.dim_edge_features == 0:
                self.convs = ModuleList([AntiSymmetricConv(in_channels=inp,
                                          num_iters=1,
                                          gamma=self.gamma,
                                          epsilon=self.epsilon,
                                          activ_fun=self.activ_fun,
                                          gcn_conv=self.gcn_norm,
                                          bias=self.bias) for _ in range(self.num_layers)])
            else:
                # DISCRETE EDGE TYPES ONLY
                self.convs = ModuleList()
                for _ in range(self.num_layers):
                    edge_convs = ModuleList()
                    for _ in range(self.dim_edge_features):
                        edge_convs.append(AntiSymmetricConv(in_channels=inp,
                                          num_iters=1,
                                          gamma=self.gamma,
                                          epsilon=self.epsilon,
                                          activ_fun=self.activ_fun,
                                          gcn_conv=self.gcn_norm,
                                          bias=self.bias))
                    self.convs.append(edge_convs)

        if not self.global_aggregation:
            self.readout = Sequential(OrderedDict([
                ('L1', Linear(inp, inp // 2)),
                ('LeakyReLU1', LeakyReLU()),
                ('L2', Linear(inp // 2, self.dim_target)),
                ('LeakyReLU2', LeakyReLU())
            ]))
        else:
            self.readout = Sequential(OrderedDict([
                ('L1', Linear(inp * 3, (inp * 3) // 2)),
                ('LeakyReLU1', LeakyReLU()),
                ('L2', Linear((inp * 3) // 2, self.dim_target)),
                ('LeakyReLU2', LeakyReLU())
            ]))

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        ## WORKS WITH DISCRETE FEATURES ONLY - implement R-GCN/GIN/ADGN
        if self.dim_edge_features > 0:
            edge_attr = data.edge_attr
            assert len(edge_attr.shape) == 1  # can only be [num_edges]

        x = self.emb(x) if self.emb else x

        if self.weight_sharing:
            if self.dim_edge_features == 0:
                x = self.conv(x, edge_index)
            else:
                outputs = 0
                for e, conv in enumerate(self.conv):
                    outputs += conv(x, edge_index[:, edge_attr == e])
                x = outputs

        else:
            if self.dim_edge_features == 0:
                for i in range(self.num_layers):
                    x = self.convs[i](x, edge_index)
            else:
                for i in range(self.num_layers):
                    outputs = 0
                    for e, conv in enumerate(self.convs[i]):
                        outputs += conv(x, edge_index[:, edge_attr == e])
                    x = outputs

        if self.global_aggregation:
            x = torch.cat(
                [global_add_pool(x, batch), global_max_pool(x, batch),
                 global_mean_pool(x, batch)], dim=1)

        x = self.readout(x)

        return x, x, [batch]