import torch
from pydgn.model.interface import ModelInterface

from torch.nn import Module, Linear, ModuleList, Sequential, LeakyReLU, GELU, \
    Dropout, ReLU
from torch.nn.functional import gelu, dropout
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_add_pool, global_max_pool, \
    global_mean_pool
from typing import Optional, Callable
from collections import OrderedDict
from torch import tanh


class DGN_GraphProp(ModelInterface):

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

        conv_layer = config.get('conv_layer', 'GCNConv')
        self.conv_name = conv_layer

        self.input_dim = dim_node_features
        self.output_dim = dim_target
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.alpha = config.get('alpha', None)

        inp = self.input_dim
        self.emb = None
        if self.hidden_dim is not None:
            self.emb = Linear(self.input_dim, self.hidden_dim)
            inp = self.hidden_dim

        if dim_edge_features == 0:
            self.conv_layer = getattr(pyg_nn, conv_layer)
            self.conv = ModuleList()
            for _ in range(self.num_layers):
                if conv_layer == 'GINConv':
                    mlp = Linear(inp, inp)
                    self.conv.append(self.conv_layer(nn=mlp,
                                                     train_eps=True))
                elif conv_layer == 'GCN2Conv':
                    self.conv.append(self.conv_layer(channels=inp,
                                                     alpha=self.alpha))
                else:
                    self.conv.append(self.conv_layer(in_channels=inp,
                                                     out_channels=inp))
        else:
            # DISCRETE EDGE TYPES ONLY
            self.conv_layer = getattr(pyg_nn, conv_layer)
            self.convs = ModuleList()
            for _ in range(self.num_layers):
                edge_convs = ModuleList()
                for _ in range(self.dim_edge_features):
                    if conv_layer == 'GINConv':
                        mlp = Linear(inp, inp)
                        edge_convs.append(self.conv_layer(nn=mlp,
                                                         train_eps=True))
                    elif conv_layer == 'GCN2Conv':
                        edge_convs.append(self.conv_layer(channels=inp,
                                                         alpha=self.alpha))
                    else:
                        edge_convs.append(self.conv_layer(in_channels=inp,
                                                         out_channels=inp))
                self.convs.append(edge_convs)

        self.node_level_task = not config['global_aggregation']

        if self.node_level_task:
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

    def forward(self, data: Data, retain_grad = False) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        ## WORKS WITH DISCRETE FEATURES ONLY - implement R-GCN/GIN/ADGN
        if self.dim_edge_features > 0:
            edge_attr = data.edge_attr
            assert len(edge_attr.shape) == 1  # can only be [num_edges]

        h_list = []

        x = self.emb(x) if self.emb else x

        if self.conv_name == 'GCN2Conv':
            x_0 = x

        if self.dim_edge_features == 0:
            for conv in self.conv:
                if retain_grad:
                    x.retain_grad()
                if self.conv_name == 'GCN2Conv':
                    x = tanh(conv(x, x_0, edge_index))
                else:
                    x = tanh(conv(x, edge_index))
                h_list.append(x)
        else:
            for edge_convs in self.convs:
                outputs = 0
                for e, conv in enumerate(edge_convs):
                    if self.conv_name == 'GCN2Conv':
                        if retain_grad:
                            x.retain_grad()

                        outputs += tanh(conv(x, x_0, edge_index[:, edge_attr == e]))
                    else:
                        if retain_grad:
                            x.retain_grad()

                        outputs += tanh(conv(x, edge_index[:, edge_attr == e]))
                x = outputs
                h_list.append(x)

        if not self.node_level_task:
            x = torch.cat(
                [global_add_pool(x, batch), global_max_pool(x, batch),
                 global_mean_pool(x, batch)], dim=1)

        x = self.readout(x)

        if not retain_grad:
            h_list = torch.stack(h_list, dim=1)

        return x, h_list, [batch]



class DGN_Peptides(ModelInterface):
    # Use some of the "tricks" in the paper https://arxiv.org/pdf/2309.00367v2.pdf

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

        conv_layer = config.get('conv_layer', 'GCNConv')
        self.conv_name = conv_layer

        self.use_encodings = config.get('use_positional_encoding', None)

        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']

        if self.use_encodings is not None:
            if self.use_encodings == 'LapPE':

                dim_pe = 16

                self.linear_A = Linear(2, 2 * dim_pe)
                self.input_expander = Linear(dim_node_features, self.hidden_dim - dim_pe)
                self.pe_encoder = Sequential(ReLU(),
                                                Linear(2 * dim_pe, 2 * dim_pe),
                                                Linear(2 * dim_pe, dim_pe),
                                                ReLU())

                dim_node_features = self.hidden_dim  # dim_node_features + dim_pe

            elif self.use_encodings == 'RWSE':
                dim_node_features = dim_node_features + 20
            else:
                raise NotImplementedError('Positional encoding option not recognized')

        self.input_dim = dim_node_features
        self.output_dim = dim_target

        inp = self.input_dim
        hd = self.hidden_dim

        self.emb = Sequential(Linear(inp, hd), GELU(), Linear(hd, hd))

        self.conv_layer = getattr(pyg_nn, conv_layer)
        self.conv = ModuleList()

        # https://github.com/toenshoff/LRGB/blob/main/graphgps/network/custom_gnn.py
        for _ in range(self.num_layers):
            if conv_layer == 'GINConv':
                mlp = Sequential((Linear(hd, hd),
                                  GELU(),
                                  Linear(hd, hd)))
                self.conv.append(self.conv_layer(nn=mlp,
                                                 train_eps=True))
            else:
                self.conv.append(self.conv_layer(in_channels=hd,
                                                 out_channels=hd))

        self.pooling = global_mean_pool # as defined in Table 1-2 in https://arxiv.org/pdf/2309.00367v2.pdf

        # 3 layers as defined in Table 1-2 in https://arxiv.org/pdf/2309.00367v2.pdf
        # https://github.com/toenshoff/LRGB/blob/main/graphgps/head/mlp_graph.py
        self.readout = Sequential(OrderedDict([
            ('Drop1', Dropout(self.dropout)),
            ('L1', Linear(hd, hd)),
            ('GeLU1', GELU()),
            ('Drop2', Dropout(self.dropout)),
            ('L2', Linear(hd, hd)),
            ('GeLU2', GELU()),
            ('Drop3', Dropout(self.dropout)),
            ('L3', Linear(hd, self.output_dim)),

        ]))

    def forward(self, data: Data, retain_grad = False) -> torch.Tensor:
        x_orig, edge_index, batch = data.x, data.edge_index, data.batch

        if self.use_encodings is not None:
            if self.use_encodings == 'LapPE':

                EigVecs = data.laplacian_eigenvector_pe
                EigVals = data.laplacian_eigenvalues_pe

                # taken from https://github.com/toenshoff/LRGB/blob/main/graphgps/encoder/laplace_pos_encoder.py
                if self.training:
                    sign_flip = torch.rand(EigVecs.size(1),
                                           device=EigVecs.device)
                    sign_flip[sign_flip >= 0.5] = 1.0
                    sign_flip[sign_flip < 0.5] = -1.0
                    EigVecs = EigVecs * sign_flip.unsqueeze(0)

                pos_enc = torch.cat((EigVecs.unsqueeze(2), EigVals),
                                    dim=2)  # (Num nodes) x (Num Eigenvectors) x 2
                empty_mask = torch.isnan(
                    pos_enc)  # (Num nodes) x (Num Eigenvectors) x 2

                pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2

                pos_enc = self.linear_A(pos_enc)  # (Num nodes) x (Num Eigenvectors) x dim_pe
                pos_enc = self.pe_encoder(pos_enc)

                # Remove masked sequences; must clone before overwriting masked elements
                pos_enc = pos_enc.clone().masked_fill_(empty_mask[:, :, 0].unsqueeze(2), 0.)

                # Sum pooling
                pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # (Num nodes) x dim_pe

                x_expanded = self.input_expander(x_orig)

                x = torch.cat((x_expanded, pos_enc), dim=1)

            elif self.use_encodings == 'RWSE':
                x = torch.cat((x_orig, data.random_walk_pe), dim=1)
        else:
            x = x_orig

        h_list = []

        x = self.emb(x)

        if retain_grad:
            x.retain_grad()

        for conv in self.conv:
            x_0 = x

            # implement a skip connection
            x = dropout(gelu(conv(x, edge_index)), p=self.dropout) + x_0

            if retain_grad:
                x.retain_grad()

            h_list.append(x)

        x_global = global_mean_pool(x, batch)

        y = self.readout(x_global)

        if not retain_grad:
            h_list = torch.stack(h_list, dim=1)

        return y, h_list, [batch]


class DGN_SingleGraph(ModelInterface):

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

        conv_layer = config.get('conv_layer', 'GCNConv')
        self.conv_name = conv_layer

        self.input_dim = dim_node_features
        self.output_dim = dim_target
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.alpha = config.get('alpha', None)

        inp = self.input_dim
        self.emb = None
        if self.hidden_dim is not None:
            self.emb = Linear(self.input_dim, self.hidden_dim)
            inp = self.hidden_dim

        if dim_edge_features == 0:
            self.conv_layer = getattr(pyg_nn, conv_layer)
            self.conv = ModuleList()
            for _ in range(self.num_layers):
                if conv_layer == 'GINConv':
                    mlp = Linear(inp, inp)
                    self.conv.append(self.conv_layer(nn=mlp,
                                                     train_eps=True))
                elif conv_layer == 'GCN2Conv':
                    self.conv.append(self.conv_layer(channels=inp,
                                                     alpha=self.alpha))
                else:
                    self.conv.append(self.conv_layer(in_channels=inp,
                                                     out_channels=inp))
        else:
            # DISCRETE EDGE TYPES ONLY
            self.conv_layer = getattr(pyg_nn, conv_layer)
            self.convs = ModuleList()
            for _ in range(self.num_layers):
                edge_convs = ModuleList()
                for _ in range(self.dim_edge_features):
                    if conv_layer == 'GINConv':
                        mlp = Linear(inp, inp)
                        edge_convs.append(self.conv_layer(nn=mlp,
                                                         train_eps=True))
                    elif conv_layer == 'GCN2Conv':
                        edge_convs.append(self.conv_layer(channels=inp,
                                                         alpha=self.alpha))
                    else:
                        edge_convs.append(self.conv_layer(in_channels=inp,
                                                         out_channels=inp))
                self.convs.append(edge_convs)

        self.node_level_task = not config['global_aggregation']

        if self.node_level_task:
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

    def forward(self, data: Data, retain_grad = False) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        training_indices, eval_indices = data.training_indices, data.eval_indices

        ## WORKS WITH DISCRETE FEATURES ONLY - implement R-GCN/GIN/ADGN
        if self.dim_edge_features > 0:
            edge_attr = data.edge_attr
            assert len(edge_attr.shape) == 1  # can only be [num_edges]

        h_list = []

        x = self.emb(x) if self.emb else x

        if self.conv_name == 'GCN2Conv':
            x_0 = x

        if self.dim_edge_features == 0:
            for conv in self.conv:
                if retain_grad:
                    x.retain_grad()
                if self.conv_name == 'GCN2Conv':
                    x = tanh(conv(x, x_0, edge_index))
                else:
                    x = tanh(conv(x, edge_index))
                h_list.append(x)
        else:
            for edge_convs in self.convs:
                outputs = 0
                for e, conv in enumerate(edge_convs):
                    if self.conv_name == 'GCN2Conv':
                        if retain_grad:
                            x.retain_grad()

                        outputs += tanh(conv(x, x_0, edge_index[:, edge_attr == e]))
                    else:
                        if retain_grad:
                            x.retain_grad()

                        outputs += tanh(conv(x, edge_index[:, edge_attr == e]))
                x = outputs
                h_list.append(x)

        if not self.node_level_task:
            x = torch.cat(
                [global_add_pool(x, batch), global_max_pool(x, batch),
                 global_mean_pool(x, batch)], dim=1)

        x = self.readout(x)


        # FOR NODE LEVEL TASKS WHERE YOU HAVE A SINGLE GRAPH
        x = x[eval_indices]
        h_list = [h[eval_indices] for h in h_list]
        batch = batch[eval_indices]

        if not retain_grad:
            h_list = torch.stack(h_list, dim=1)

        y = data.y[eval_indices]
        return x, h_list, [batch, y]
