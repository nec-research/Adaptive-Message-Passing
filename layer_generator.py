from collections import OrderedDict
from typing import Callable, Tuple, Optional

import torch
from torch import nn, relu, Tensor, tanh
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, Module, ModuleList, \
    Dropout, GELU
from torch_geometric.nn import GINConv, GCNConv, GINEConv, ResGatedGraphConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import OptPairTensor
from torch_sparse import SparseTensor

from adgn import AntiSymmetricConv


class EdgeFilterGINConv(GINConv):

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_filter: Optional[torch.Tensor] = None,
        size = None,
        activation=torch.nn.functional.tanh
    ) -> torch.Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x,
                             edge_weight=edge_weight,
                             edge_filter=edge_filter,
                             size=None)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return activation(self.nn(out))

    # def message(
    #     self,
    #     x_j: torch.Tensor,
    #     edge_weight: Optional[torch.Tensor] = None,
    #     edge_filter: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #
    #     # backward compatibility
    #     if edge_filter is not None and len(edge_filter.shape) == 1:
    #         edge_filter = edge_filter.view(-1, 1)
    #
    #     if edge_filter is not None:
    #         return edge_filter * x_j
    #     else:
    #         return x_j

    def message(
        self,
        x_j: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_filter: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # backward compatibility
        if edge_filter is not None and len(edge_filter.shape) == 1:
            edge_filter = edge_filter.view(-1, 1)

        if edge_weight is None and edge_filter is not None:
            return edge_filter * x_j

        elif edge_weight is None and edge_filter is None:
            return x_j

        elif edge_weight is not None and edge_filter is None:
            return edge_weight.view(-1, 1) * x_j

        else:
            return edge_filter * edge_weight.view(-1, 1) * x_j



class EdgeFilterGINEConv(GINEConv):

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_filter: Optional[torch.Tensor] = None,
        size = None,
        activation=torch.nn.functional.tanh
    ) -> torch.Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x,
                             edge_weight=edge_weight,
                             edge_filter=edge_filter,
                             size=None)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return activation(self.nn(out))

    def message(
        self,
        x_j: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_filter: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if self.lin is not None:
            edge_weight = self.lin(edge_weight.float())

        # backward compatibility
        if edge_filter is not None and len(edge_filter.shape) == 1:
            edge_filter = edge_filter.view(-1, 1)

        if edge_weight is None and edge_filter is not None:
            return edge_filter * x_j

        elif edge_weight is None and edge_filter is None:
            return x_j

        elif edge_weight is not None and edge_filter is None:
            return edge_weight + x_j
        else:
            return edge_filter * (edge_weight + x_j)


class EdgeFilterGatedGCNConv(ResGatedGraphConv):

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_filter: Optional[torch.Tensor] = None,
        size = None,
        activation=torch.nn.functional.tanh
    ) -> torch.Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        k = self.lin_key(x[1])
        q = self.lin_query(x[0])
        v = self.lin_value(x[0])

        # propagate_type: (k: Tensor, q: Tensor, v: Tensor)
        out = self.propagate(edge_index, k=k, q=q, v=v,
                             edge_weight=None,
                             edge_filter=edge_filter)

        if self.root_weight:
            out = out + self.lin_skip(x[1])

        if self.bias is not None:
            out = out + self.bias

        return activation(out)

    # def message(
    #     self,
    #     k_i: Tensor,
    #     q_j: Tensor,
    #     v_j: Tensor,
    #     edge_weight: Optional[torch.Tensor] = None,
    #     edge_filter: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #
    #     x_j = self.act(k_i + q_j) * v_j  # the node message
    #
    #     # backward compatibility
    #     if edge_filter is not None and len(edge_filter.shape) == 1:
    #         edge_filter = edge_filter.view(-1, 1)
    #
    #     if edge_filter is not None:
    #         return edge_filter * x_j
    #     else:
    #         return x_j

    def message(
        self,
        k_i: Tensor,
        q_j: Tensor,
        v_j: Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_filter: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        x_j = self.act(k_i + q_j) * v_j  # the node message

        # backward compatibility
        if edge_filter is not None and len(edge_filter.shape) == 1:
            edge_filter = edge_filter.view(-1, 1)

        if edge_weight is None and edge_filter is not None:
            return edge_filter * x_j

        elif edge_weight is None and edge_filter is None:
            return x_j
        elif edge_weight is not None and edge_filter is None:
            return edge_weight.view(-1, 1) * x_j

        else:
            return edge_filter * edge_weight.view(-1, 1) * x_j



class EdgeFilterGCNConv(GCNConv):

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_filter: Optional[torch.Tensor] = None,
        activation=torch.nn.functional.tanh
    ) -> torch.Tensor:

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow,
                        x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow,
                        x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x,
                             edge_weight=edge_weight,
                             edge_filter=edge_filter,
                             size=None)

        if self.bias is not None:
            out = out + self.bias

        return activation(out)

    def message(
        self,
        x_j: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_filter: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # backward compatibility
        if edge_filter is not None and len(edge_filter.shape) == 1:
            edge_filter = edge_filter.view(-1, 1)

        if edge_weight is None and edge_filter is not None:
            return edge_filter * x_j

        elif edge_weight is None and edge_filter is None:
            return x_j

        elif edge_weight is not None and edge_filter is None:
            return edge_weight.view(-1, 1) * x_j

        else:
            return edge_filter * edge_weight.view(-1, 1) * x_j



class BottleneckCNN(nn.Module):
    """Construction block for the CNN"""

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, residual=True):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            self.expansion * out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.is_residual = residual
        self.shortcut = nn.Sequential()
        if (
            stride != 1 or in_channels != self.expansion * out_channels
        ) and self.is_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * out_channels),
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.is_residual:
            out += self.shortcut(x)
        out = relu(out)
        return out


class LayerGenerator:
    def __init__(self, **kwargs):
        super(LayerGenerator, self).__init__()

    def make_generators(
        self, node_size, edge_size, hidden_size, output_size, **kwargs
    ) -> Tuple[Callable, Callable]:
        """
        Creates two hidden and output layer generators using the
        provided parameters. They both accept a layer id. If the layer id is 0,
        it is assumed that we are generating the layers for the input
        """
        raise NotImplementedError("To be implemented in a sub-class")


class CifarGenerator(LayerGenerator):
    """Adapted from UDN paper's source code"""

    def make_generators(
        self, node_size, edge_size, hidden_size, output_size, **kwargs
    ) -> Tuple[Callable, Callable]:
        def make_hidden(layer_id: int):
            if layer_id == -1:
                return None, 3, 32

            # torch.manual_seed(layer_id)
            # np.random.seed(layer_id)

            FIRST_LAYER_CHANNEL = 64
            out_dim = 32
            if layer_id == 0:
                bloc = [
                    nn.Conv2d(
                        3,
                        FIRST_LAYER_CHANNEL,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(FIRST_LAYER_CHANNEL),
                    nn.ReLU(),
                ]
                bloc = nn.Sequential(*bloc)

                # print(layer_id, 'hidden')
                # print([p for p in bloc.parameters()])
                # total_params = sum(p.sum().item() for p in bloc.parameters())
                # print(f"Hidden Sum parameters value layer id {layer_id}: {total_params}")

                return bloc, FIRST_LAYER_CHANNEL, out_dim

            in_channel = FIRST_LAYER_CHANNEL
            total = 0
            for block_id in range(1, 4):
                n_of_blocks = [0, 3, 5, 1000][block_id]
                channels_of_block = 2 ** (block_id + 5)

                for i in range(n_of_blocks):
                    total += 1
                    if i == 0:
                        stride = 2
                    else:
                        stride = 1

                    out_channel = channels_of_block * BottleneckCNN.expansion
                    out_dim //= stride

                    if total == layer_id:
                        bloc = BottleneckCNN(
                            in_channel,
                            channels_of_block,
                            stride,
                            residual=True,
                        )

                        # print(layer_id)
                        # print([p for p in bloc.parameters()])
                        # total_params = sum(
                        #     p.sum().item() for p in bloc.parameters())
                        # print(
                        #     f"Hidden Sum parameters value layer id {layer_id}: {total_params}")

                        return bloc, out_channel, out_dim

                    in_channel = out_channel

        def make_output(layer_id: int):
            # if layer_id == -1:
            #     torch.manual_seed(0)
            #     np.random.seed(0)
            # else:
            #     torch.manual_seed(layer_id)
            #     np.random.seed(layer_id)

            _, last_channels, last_dim = make_hidden(layer_id)

            last_hidden_size = (last_dim // 4) ** 2 * last_channels

            layers = [
                nn.AvgPool2d(4),
                nn.Flatten(),
                nn.Linear(last_hidden_size, 10),
            ]
            l = nn.Sequential(*layers)

            # print(layer_id)
            # print([p for p in l.parameters()]);
            # exit(0)

            # total_params = sum(p.sum().item() for p in l.parameters())
            # print(f"Output Sum parameters value layer id {layer_id}: {total_params}")

            return l

        return lambda layer_id: make_hidden(layer_id)[0], make_output


class MLPGenerator(LayerGenerator):
    def make_generators(
        self, node_size, edge_size, hidden_size, output_size, **kwargs
    ) -> Tuple[Callable, Callable]:
        def make_hidden(layer_id: int):
            assert layer_id >= 0

            if layer_id == 0:
                return Sequential(Linear(node_size, hidden_size), ReLU())
            else:
                return Sequential(Linear(hidden_size, hidden_size), ReLU())

        def make_output(layer_id: int):
            assert layer_id >= 0

            if layer_id == 0:
                return Linear(node_size, output_size)
            else:
                return Linear(hidden_size, output_size)

        return make_hidden, make_output


class RelationalConv(Module):
    """
    Wrapper that implements multiple convolutions at a given layer,
    one for each DISCRETE edge type. Breaks if continuous values are used
    """

    def __init__(self, edge_size, conv_layer, **kwargs):
        super(RelationalConv, self).__init__()
        self.edge_size = edge_size
        self.edge_convs = ModuleList()

        for _ in range(edge_size):
            self.edge_convs.append(conv_layer(**kwargs))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_filter: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = 0

        for e, conv in enumerate(self.edge_convs):
            if edge_filter is None:
                outputs += tanh(conv(x, edge_index[:, edge_attr == e]))
            else:
                outputs += tanh(conv(x, edge_index[:, edge_attr == e],
                                     edge_filter=edge_filter[edge_attr == e]))

        return outputs


class ADGNGenerator(LayerGenerator):
    def make_generators(
        self,
        node_size,
        edge_size,
        hidden_size,
        output_size,
        **kwargs,
    ) -> Tuple[Callable, Callable]:
        def make_hidden(layer_id: int):
            if layer_id == 0:
                return Linear(node_size, hidden_size)
            else:
                if edge_size == 0:
                    # stacking Antisymmetric DGN convolutions rather than weight
                    # sharing (see Antisymmetric DGN paper)
                    return AntiSymmetricConv(
                        in_channels=hidden_size,
                        num_iters=1,
                        epsilon=kwargs["adgn_epsilon"],
                        gamma=kwargs["adgn_gamma"],
                        bias=kwargs["adgn_bias"],
                        gcn_conv=kwargs["adgn_gcn_norm"],
                        activ_fun=kwargs["adgn_activ_fun"],
                    )
                else:
                    # DISCRETE EDGE ONLY
                    return RelationalConv(edge_size=edge_size,
                                          conv_layer=AntiSymmetricConv,
                                          in_channels=hidden_size,
                                          num_iters=1,
                                          epsilon=kwargs["adgn_epsilon"],
                                          gamma=kwargs["adgn_gamma"],
                                          bias=kwargs["adgn_bias"],
                                          gcn_conv=kwargs["adgn_gcn_norm"],
                                          activ_fun=kwargs["adgn_activ_fun"])

        def make_output(layer_id: int):

            if not kwargs["global_aggregation"]:
                if layer_id == -1:
                    return Sequential(OrderedDict([
                                ('L1', Linear(node_size, node_size // 2)),
                                ('LeakyReLU1', LeakyReLU()),
                                ('L2', Linear(node_size // 2, output_size)),
                                ('LeakyReLU2', LeakyReLU())
                    ]))
                else:
                    return Sequential(OrderedDict([
                                ('L1', Linear(hidden_size, hidden_size // 2)),
                                ('LeakyReLU1', LeakyReLU()),
                                ('L2', Linear(hidden_size // 2, output_size)),
                                ('LeakyReLU2', LeakyReLU())
                    ]))
            else:
                if layer_id == -1:
                    return Sequential(OrderedDict([
                                ('L1', Linear((node_size*3), (node_size*3) // 2)),
                                ('LeakyReLU1', LeakyReLU()),
                                ('L2', Linear((node_size*3) // 2, output_size)),
                                ('LeakyReLU2', LeakyReLU())
                    ]))
                else:
                    return Sequential(OrderedDict([
                                ('L1', Linear((hidden_size*3), (hidden_size*3) // 2)),
                                ('LeakyReLU1', LeakyReLU()),
                                ('L2', Linear((hidden_size*3) // 2, output_size)),
                                ('LeakyReLU2', LeakyReLU())
                    ]))



        return make_hidden, make_output


class DGNGenerator(LayerGenerator):
    def make_generators(
        self,
        node_size,
        edge_size,
        hidden_size,
        output_size,
        **kwargs,
    ) -> Tuple[Callable, Callable]:

        def make_hidden(layer_id: int):
            if layer_id == 0:
                return Linear(node_size, hidden_size)
            else:
                conv_name = kwargs['conv_layer']

                if edge_size == 0:

                    if conv_name == 'GINConv':
                        mlp = Linear(hidden_size, hidden_size)
                        return EdgeFilterGINConv(nn=mlp, train_eps=True)

                    elif conv_name == 'GCNConv':
                        return EdgeFilterGCNConv(in_channels=hidden_size,
                                                 out_channels=hidden_size,
                                                 add_self_loops=False)
                    else:
                        raise NotImplementedError(f'Conv layer not recognized: {conv_name}')
                else:
                    # DISCRETE EDGE ONLY
                    if conv_name == 'GINConv':
                        mlp = Linear(hidden_size, hidden_size)
                        return RelationalConv(edge_size=edge_size,
                                              conv_layer=EdgeFilterGINConv,
                                              nn=mlp, train_eps=True)

                    elif conv_name == 'GCNConv':
                        return RelationalConv(edge_size=edge_size,
                                              conv_layer=EdgeFilterGCNConv,
                                              in_channels=hidden_size,
                                              out_channels=hidden_size,
                                              add_self_loops=False)
                    else:
                        raise NotImplementedError(
                            f'Conv layer not recognized: {conv_name}')

        def make_output(layer_id: int):

            if not kwargs["global_aggregation"]:
                if layer_id == -1:
                    return Sequential(OrderedDict([
                                ('L1', Linear(node_size, node_size // 2)),
                                ('LeakyReLU1', LeakyReLU()),
                                ('L2', Linear(node_size // 2, output_size)),
                                ('LeakyReLU2', LeakyReLU())
                    ]))
                else:
                    return Sequential(OrderedDict([
                                ('L1', Linear(hidden_size, hidden_size // 2)),
                                ('LeakyReLU1', LeakyReLU()),
                                ('L2', Linear(hidden_size // 2, output_size)),
                                ('LeakyReLU2', LeakyReLU())
                    ]))
            else:
                if layer_id == -1:
                    return Sequential(OrderedDict([
                                ('L1', Linear((node_size*3), (node_size*3) // 2)),
                                ('LeakyReLU1', LeakyReLU()),
                                ('L2', Linear((node_size*3) // 2, output_size)),
                                ('LeakyReLU2', LeakyReLU())
                    ]))
                else:
                    return Sequential(OrderedDict([
                                ('L1', Linear((hidden_size*3), (hidden_size*3) // 2)),
                                ('LeakyReLU1', LeakyReLU()),
                                ('L2', Linear((hidden_size*3) // 2, output_size)),
                                ('LeakyReLU2', LeakyReLU())
                    ]))

        return make_hidden, make_output


class LRGBGenerator(LayerGenerator):
    def make_generators(
        self,
        node_size,
        edge_size,
        hidden_size,
        output_size,
        **kwargs,
    ) -> Tuple[Callable, Callable]:

        dropout = kwargs['dropout']

        def make_hidden(layer_id: int):
            if layer_id == 0:
                return Sequential(Linear(node_size, hidden_size),
                                  GELU(),
                                  Linear(hidden_size, hidden_size))
            else:
                conv_name = kwargs['conv_layer']

                if conv_name == 'GINConv':
                    mlp = Sequential(Linear(hidden_size, hidden_size),
                                      GELU(),
                                      Linear(hidden_size, hidden_size))
                    return EdgeFilterGINConv(nn=mlp, train_eps=True)

                elif conv_name == 'GCNConv':
                    return EdgeFilterGCNConv(in_channels=hidden_size,
                                             out_channels=hidden_size,
                                             add_self_loops=False)
                elif conv_name == 'GINEConv':
                    mlp = Sequential(Linear(hidden_size, hidden_size),
                                      GELU(),
                                      Linear(hidden_size, hidden_size))
                    return EdgeFilterGINEConv(nn=mlp,
                                              train_eps=True,
                                              edge_dim=edge_size)
                elif conv_name == 'ResGatedGraphConv':
                    return EdgeFilterGatedGCNConv(hidden_size, hidden_size)
                else:
                    raise NotImplementedError(f'Conv layer not recognized: {conv_name}')


        def make_output(layer_id: int):

            if layer_id == -1:
                return Sequential(
                            Linear(node_size, hidden_size),
                            GELU(),
                            Dropout(dropout),
                            Linear(hidden_size, hidden_size),
                            GELU(),
                            Dropout(dropout),
                            Linear(hidden_size, output_size),
                )
            else:
                return Sequential(
                            Dropout(dropout),
                            Linear(hidden_size, hidden_size),
                            GELU(),
                            Dropout(dropout),
                            Linear(hidden_size, hidden_size),
                            GELU(),
                            Dropout(dropout),
                            Linear(hidden_size, output_size),
                )


        return make_hidden, make_output