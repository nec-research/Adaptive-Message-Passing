from typing import Tuple, Optional, List, Callable, Mapping, Any

import numpy as np
import torch
from pydgn.evaluation.util import return_class_and_args
from pydgn.model.interface import ModelInterface
from pydgn.training.callback.optimizer import Optimizer
from sklearn.metrics import accuracy_score
from torch import sigmoid
from torch.distributions import Normal
from torch.nn import (
    ModuleList,
    CrossEntropyLoss,
    Linear,
    Sequential,
    ReLU,
    Tanh,
    Softplus,
    Module,
)
from torch.nn.functional import softplus, dropout, gelu
from torch.nn.parameter import Parameter
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_mean_pool, \
    global_max_pool

from distribution import (
    DiscretizedDistribution,
    TruncatedDistribution,
    ContinuousDistribution,
    MixtureTruncated, FixedDepth,
)
from util import (
    count_parameters,
    update_optimizer,
    get_current_parameters,
    add_new_params_to_optimizer,
)


class FilterNetwork(Module):
    def __init__(self, input_features, config):
        super(FilterNetwork, self).__init__()

        self.input_features = input_features
        self.hidden_dim = config["hidden_dim"]

        self.filter_type = config['filter_messages']
        self.message_filters = ModuleList(
            [
                Sequential(
                    Linear(input_features, self.hidden_dim),
                    Tanh(),
                    Linear(self.hidden_dim, self.hidden_dim),
                )
            ]
        )

    def to(self, device):
        """Set the device of the model."""
        super().to(device)
        self.device = device

    def forward(self, current_state, current_layer):
        return sigmoid(self.message_filters[current_layer](current_state))

    def add_one_layer_to_filter(self, torch_optimizer):
        if self.filter_type == "input_features":
            # same logic of UDN
            self.message_filters.append(
                Sequential(
                    Linear(self.input_features, self.hidden_dim),
                    Tanh(),
                    Linear(self.hidden_dim, self.hidden_dim),
                ).to(self.device)
            )
        else:
            # same logic of UDN
            self.message_filters.append(
                Sequential(
                    Linear(self.hidden_dim, self.hidden_dim),
                    Tanh(),
                    Linear(self.hidden_dim, self.hidden_dim),
                ).to(self.device)
            )

        if self.training and torch_optimizer is not None:
            # update the meta information in the optimizer with the new params
            torch_optimizer.param_groups[0]["params"].extend(
                self.message_filters[-1].parameters()
            )


class UnboundedDepthNetwork(ModelInterface):
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

        # hidden layer size
        self.hidden_dim = config["hidden_dim"]

        # for LRGB datasets
        self.use_encodings = config.get('use_positional_encoding', None)
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

        # to be set up later by the PyDGN engine
        self.torch_optimizer = None

        # to be set up after initialization of parameters
        self.current_depth = None

        self.n_obs = config["n_observations"]

        # store the quantile we want to use
        self.quantile = config["quantile"]

        self.conv_layer = config.get('conv_layer', '')

        # instantiate hidden and output layer generator
        l_gen_cls, l_gen_args = return_class_and_args(
            config, "layer_generator"
        )
        self.layer_generator = l_gen_cls(**l_gen_args)
        (
            self.hidden_generator,
            self.output_generator,
        ) = self.layer_generator.make_generators(
            dim_node_features,
            dim_edge_features,
            self.hidden_dim,
            dim_target,
            **self.config
        )

        # these lists of layers will be dynamically updated
        self.hidden_layers = ModuleList([])

        # create the very first mapping from input to output
        self.output_layers = ModuleList([self.output_generator(layer_id=-1)])

        t_dist_cls, t_dist_args = return_class_and_args(
            config, "truncated_distribution"
        )
        if t_dist_cls in [TruncatedDistribution, FixedDepth]:
            truncated_dist = t_dist_cls(
                truncation_quantile=self.quantile, **t_dist_args
            )
        elif t_dist_cls == MixtureTruncated:
            list_distr = []
            kwargs = {}

            for k in t_dist_args.keys():
                if "discretized_distribution" in k:
                    list_distr.append({k: t_dist_args[k]})
                else:
                    kwargs[k] = t_dist_args[k]

            truncated_dist = t_dist_cls(
                truncation_quantile=self.quantile,
                distribution_list=list_distr,
                **kwargs
            )
        self.variational_L = truncated_dist

        # # Instantiate the variational distribution q(\theta | \ell)
        # NOTE: not needed, see comment in forward method
        # q_theta_L_cls, q_theta_L_args = s2c(config['q_theta_given_L'])
        # q_theta_L = q_theta_L_cls(q_theta_L_args)
        # self.variational_theta = q_theta_L

        # prior scale for p(theta) - we use a normal with mean 0
        self.theta_prior_scale = config["theta_prior_scale"]

        # prior mean and scale for p(ell)

        l_prior_cls, l_prior_args = return_class_and_args(
            config, "layer_prior"
        )
        if l_prior_cls is not None:
            self.layer_prior = l_prior_cls(**l_prior_args)
        else:
            # uninformative prior
            self.layer_prior = None

        self.filter_messages = config.get("filter_messages", None)

        if self.filter_messages:
            self.filter_network = FilterNetwork(dim_node_features,  config)
        else:
            self.filter_network = None

        self.return_fake_embeddings = config.get(
            "return_fake_embeddings", False
        )

        self.global_aggregation = self.config.get('global_aggregation', False)

        self.device = None

    def to(self, device):
        """Set the device of the model."""
        super(UnboundedDepthNetwork, self).to(device)

        self.device = device

        if self.filter_network:
            self.filter_network.to(device)
        self.variational_L.to(device)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True
    ):
        max_l = 10000

        if self.current_depth is None:
            self.current_depth = 1

        for i in range(self.current_depth, max_l):
            try:
                self.update_depth(force_depth=i)
                super().load_state_dict(state_dict, strict)

                # executed only if exception is not raised
                self.current_depth = i
                break

            except Exception as e:
                pass

        if i == max_l:
            raise Exception("Something went wrong with checkpointing")

        # print(f'Checkpoint loaded, {self.current_depth} layers')

    def update_depth(self, force_depth: int = -1):
        """
        Compute the current maximal depth of the variational
        posterior q(L) and create new layers if needed.

        Adapted from original paper (see header above)
        """
        assert self.device is not None, "Device has not been set"

        if force_depth == -1:
            self.current_depth = self.variational_L.compute_truncation_number()
        else:
            # used for checkpointing
            self.current_depth = force_depth

        assert self.current_depth > 0

        while self.current_depth > len(self.hidden_layers):
            hidden_layer = self.hidden_generator(len(self.hidden_layers))
            """
             we virtually just added a hidden_layer, so use +1
             this assumes that we initialize a first output layer in the
             init call
            """
            output_layer = self.output_generator(len(self.hidden_layers))

            hidden_layer.to(self.device)
            output_layer.to(self.device)

            self.hidden_layers.append(hidden_layer)
            self.output_layers.append(output_layer)

            if self.training and self.torch_optimizer is not None:
                # update the meta information in the optimizer with the new params
                self.torch_optimizer.param_groups[0]["params"].extend(
                    self.hidden_layers[-1].parameters()
                )

                self.torch_optimizer.param_groups[0]["params"].extend(
                    self.output_layers[-1].parameters()
                )

            # ADAPTATION OF MESSAGE FILTER LAST LAYER
            # see https://avalanche-api.continualai.org/en/v0.4.0/_modules/avalanche/models/dynamic_modules.html#IncrementalClassifier
            if self.filter_messages:
                self.filter_network.add_one_layer_to_filter(
                    self.torch_optimizer
                )

    def set_optimizer(self, optimizer: Optimizer):
        """
        Set the optimizer to later add the dynamically created
           layers' parameters to it.
        """
        # recover torch Optimizer object from PyDGN one
        self.torch_optimizer = optimizer.optimizer

    def get_q_ell_named_parameters(self) -> dict:
        return self.variational_L.get_q_ell_named_parameters()

    def _hidden_forward(
        self,
        hidden_layer,
        current_state,
        edge_index,
        edge_attr,
        edge_msg_filter,
        data,
        layer_id,
    ):
        raise NotImplementedError("To be implemented in a subclass")

    def _output_forward(
        self,
        output_layer,
        current_state,
        edge_index,
        edge_attr,
        data,
        layer_id,
    ):
        raise NotImplementedError("To be implemented in a subclass")

    def forward(
        self, data: Batch, retain_grad=False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        # first, determine if new layers have to be added
        self.update_depth()

        x_orig, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

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

        # computes probability vector of variational distr. q(layer)
        qL_probs = self.variational_L.compute_probability_vector()

        # ---- COMPUTE MESSAGE FILTERING DISTRIBUTION --- #

        message_filters_list = []

        # necessary to apply the filter network in the case of
        # 'input_features'
        first_state = x

        current_state = x

        # ----------------------------------------------- #

        hidden_state_list = []
        output_state_list = []

        log_p_theta_hidden_list = []
        log_p_theta_output_list = []
        log_p_L_list = []

        predictions_per_layer = []

        log_theta_hidden_cumulative = torch.tensor([0.0], device=self.device)
        log_theta_output_cumulative = torch.tensor([0.0], device=self.device)

        assert qL_probs.shape[0] == self.current_depth + 1

        # ----------------------------------------------- #

        for l in range(qL_probs.shape[0]):
            # ---- COMPUTE MESSAGE FILTERING DISTRIBUTION --- #

            # next message will be remodulated when propagated to neighbors
            if self.filter_network is not None and l > 0:

                if self.filter_messages == "input_features":
                    message_filters = self.filter_network(
                        first_state, l - 1
                    )  # nodes x m(q)
                else:
                    message_filters = self.filter_network(
                        current_state, l - 1
                    )  # nodes x m(q)

                message_filters_list.append(message_filters)

            # ----------------------------------------------- #

            # NOTE: assuming source_to_target flow of messages
            if self.filter_network is not None and l > 0:
                # assert message_filters.shape[1] == 1, message_filters.shape
                edge_msg_filter = message_filters[edge_index[0]]
            else:
                edge_msg_filter = None

            # computes log(p(theta)) for the hidden layers
            # assuming mu = 0 and a first order approximation
            if l > 0:
                hidden_layer = self.hidden_layers[l - 1]

                # Sum the parameters
                # total_sum = 0
                # for name, param in hidden_layer.named_parameters():
                #     total_sum += param.sum()
                #
                # print(f'total for hidden layer {l} is {total_sum}')

                # i-1 because we always have the first additional input layer
                if retain_grad:
                    current_state.retain_grad()

                current_state = self._hidden_forward(
                    hidden_layer,
                    current_state,
                    edge_index,
                    edge_attr,
                    edge_msg_filter,
                    data,
                    l,
                )

                # log_p_theta_hidden = sum([self.prior_theta.log_prob(p).sum() for p in hidden_layer.parameters()])
                if self.theta_prior_scale is not None:

                    log_p_theta_hidden = sum(
                        [
                            -(p**2).sum() / 2 / (self.theta_prior_scale**2)
                            for p in hidden_layer.parameters()
                        ]
                    ).unsqueeze(0)
                else:
                    log_p_theta_hidden = torch.tensor(0.0, device=self.device)

            # don't do anything at the first layer
            else:
                log_p_theta_hidden = torch.tensor(0.0, device=self.device)

            # do the same for the output parameters (can set theta_prior_scale to a very large number)
            output_layer = self.output_layers[l]

            # Sum the parameters
            # total_sum = 0
            # for name, param in output_layer.named_parameters():
            #     total_sum += param.sum()
            #
            # print(f'total for output layer {l} is {total_sum}')

            # computes log(p(theta)) for the output layers
            # assuming mu = 0 and a first order approximation
            if self.theta_prior_scale is not None:
                log_p_theta_output = sum(
                    [
                        -(p**2).sum() / 2 / (self.theta_prior_scale**2)
                        for p in output_layer.parameters()
                    ]
                ).unsqueeze(0)
            else:
                log_p_theta_output = torch.tensor(0.0, device=self.device)

            # compute log(p(ell))
            if self.layer_prior is not None and l > 0.0:
                log_p_L = self.layer_prior.log_prob(
                    torch.tensor([l - 1.0])
                ).to(self.device)
            else:
                log_p_L = torch.zeros(1).to(self.device)

            # compound hidden prior probs of each layer
            # due to the double summation
            log_theta_hidden_cumulative += log_p_theta_hidden

            # This was commented in the original code (possibly) because the
            # weights in UDN are shared for the output (same parameters for all
            # output layers)?.,
            log_theta_output_cumulative += log_p_theta_output

            current_output = self._output_forward(
                output_layer, current_state, edge_index, edge_attr, data, l
            )

            if l > 0 and not self.return_fake_embeddings:
                hidden_state_list.append(current_state)
            output_state_list.append(current_output)
            log_p_theta_hidden_list.append(log_theta_hidden_cumulative.clone())

            log_p_theta_output_list.append(log_theta_output_cumulative.clone())

            log_p_L_list.append(log_p_L)

        # Note: since the assumption that q(theta; nu) = N(theta; nu, I), it
        # follows that log(N(nu; nu, I)) is a constant number and can be
        # avoided in the optimization process

        # compute -\sum_ell q(ell)log(q(ell)
        # pass (un)normalized log probabilities
        entropy_qL = (
            torch.distributions.Categorical(probs=qL_probs[1:])
            .entropy()
            .unsqueeze(0)
        )  # shape [1]

        # Create batches
        # do not keep the intermediate state list because with resnet the
        # dimensions change across layers
        if l > 0 and not self.return_fake_embeddings:
            if not retain_grad:
                hidden_state_list = torch.stack(
                    hidden_state_list, dim=1
                )  # ? x depth
        output_state_list = torch.stack(output_state_list, dim=1)  # ? x depth

        log_p_theta_hidden_list = torch.stack(
            log_p_theta_hidden_list, dim=1
        )  # 1 x depth
        log_p_theta_output_list = torch.stack(
            log_p_theta_output_list, dim=1
        )  # 1 x depth
        log_p_L_list = torch.stack(log_p_L_list, dim=1)  # 1 x depth
        qL_probs = qL_probs.unsqueeze(0)  # 1 x depth

        if self.filter_network is not None:
            message_filters_list = torch.stack(message_filters_list, dim=1)
        else:
            message_filters_list = None

        return (
            output_state_list,
            hidden_state_list
            if not self.return_fake_embeddings
            else torch.zeros(output_state_list.shape[0], 1),
            (
                data.batch,
                log_p_theta_hidden_list,
                log_p_theta_output_list,
                log_p_L_list,
                entropy_qL,
                qL_probs,
                self.n_obs,
                message_filters_list
            ),
        )


class UDN_FlatInput(UnboundedDepthNetwork):
    def _hidden_forward(
        self,
        hidden_layer,
        current_state,
        edge_index,
        edge_attr,
        _,
        data,
        layer_id,
    ):
        return hidden_layer(current_state)

    def _output_forward(
        self,
        output_layer,
        current_state,
        edge_index,
        edge_attr,
        data,
        layer_id,
    ):
        return output_layer(current_state)


class AMP(UnboundedDepthNetwork):
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

    def _hidden_forward(
        self,
        hidden_layer,
        current_state,
        edge_index,
        edge_attr,
        edge_msg_filter,
        data,
        layer_id,
    ):
        assert layer_id > 0

        if layer_id == 1:
            return hidden_layer(current_state)
        else:
            return hidden_layer(
                current_state, edge_index, edge_attr, edge_msg_filter
            )

    def _output_forward(
        self,
        output_layer,
        current_state,
        edge_index,
        edge_attr,
        data,
        layer_id,
    ):
        if self.global_aggregation:
            tmp_output = torch.cat(
                [global_add_pool(current_state, data.batch),
                 global_max_pool(current_state, data.batch),
                 global_mean_pool(current_state, data.batch)], dim=1)
        else:
            tmp_output = current_state

        return output_layer(tmp_output)


class AMP_LRGB(UnboundedDepthNetwork):
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
        self.dropout = config['dropout']

    def _hidden_forward(
        self,
        hidden_layer,
        current_state,
        edge_index,
        edge_attr,
        edge_msg_filter,
        data,
        layer_id,
    ):
        assert layer_id > 0

        if layer_id == 1:
            x = hidden_layer(current_state)
        else:
            # x_0 = current_state
            x = hidden_layer(
                current_state, edge_index, edge_attr if self.conv_layer == 'GINEConv' else None, edge_msg_filter,
                activation=gelu
            )

            # implement a skip connection
            # x = dropout(x, p=self.dropout) + x_0
            # x = dropout(x, p=self.dropout)

        return x


    def _output_forward(
        self,
        output_layer,
        current_state,
        edge_index,
        edge_attr,
        data,
        layer_id,
    ):
        return output_layer(global_mean_pool(current_state, data.batch))