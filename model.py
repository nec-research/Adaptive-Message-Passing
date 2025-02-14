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
from torch.nn.functional import softplus
from torch.nn.parameter import Parameter
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_mean_pool, \
    global_max_pool

from distribution import (
    DiscretizedDistribution,
    TruncatedDistribution,
    ContinuousDistribution,
    MixtureTruncated,
)
from util import (
    count_parameters,
    update_optimizer,
    get_current_parameters,
    add_new_params_to_optimizer,
)


class FilterNetwork(Module):
    def __init__(self, input_features, output_features, config):
        super(FilterNetwork, self).__init__()

        self.config = config

        self.input_features = input_features
        self.output_features = output_features
        self.hidden_dim = config["hidden_dim"]

        self.filter_type = config["filter_messages"]

        if self.filter_type == "input_features":
            self.message_filter_hidden = Sequential(
                Linear(input_features, self.hidden_dim), Tanh()
            )
            self.message_filter_output = Linear(self.hidden_dim, 1)
        elif self.filter_type == "embedding-weight-sharing":
            self.message_filter_input = Sequential(
                Linear(input_features, self.hidden_dim), Tanh()
            )

            self.message_filter_hidden = Sequential(
                Linear(self.hidden_dim, self.hidden_dim), Tanh()
            )
            self.message_filter_output = Linear(self.hidden_dim, 1)

        elif self.filter_type == "embedding-no-weight-sharing":
            self.message_filters = ModuleList(
                [
                    Sequential(
                        Linear(input_features, self.hidden_dim),
                        Tanh(),
                        Linear(self.hidden_dim, 1),
                    )
                ]
            )
        else:
            raise NotImplementedError("error, filter type not recognized")

    def to(self, device):
        """Set the device of the model."""
        super().to(device)

        self.device = device

    def forward(self, current_state, current_layer=None):
        if self.filter_type == "input_features":
            tmp = self.message_filter_hidden(current_state)
            return sigmoid(self.message_filter_output(tmp))
        elif self.filter_type == "embedding-weight-sharing":
            if current_layer == 0:
                tmp = self.message_filter_input(current_state)
            else:
                tmp = self.message_filter_hidden(current_state)
            return sigmoid(self.message_filter_output(tmp))
        else:
            assert current_layer is not None
            return sigmoid(self.message_filters[current_layer](current_state))

    def add_one_layer_to_filter(self, torch_optimizer):

        if self.filter_type in ["input_features", "embedding-weight-sharing"]:
            old_nlayers = self.message_filter_output.out_features
            old_w = self.message_filter_output.weight
            old_b = self.message_filter_output.bias

            optimized_params = get_current_parameters(self)

            new_message_filter_output = Linear(
                self.hidden_dim, old_nlayers + 1
            ).to(self.device)
            new_message_filter_output.weight.data[:old_nlayers] = old_w
            new_message_filter_output.bias.data[:old_nlayers] = old_b

            self.message_filter_output = new_message_filter_output

            new_params = get_current_parameters(self)

            if self.training:
                update_optimizer(
                    torch_optimizer, new_params, optimized_params, reset_state=True
                )
        else:
            # same logic of UDN
            self.message_filters.append(
                Sequential(
                    Linear(self.hidden_dim, self.hidden_dim),
                    Tanh(),
                    Linear(self.hidden_dim, 1),
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
        # to be set up later by the PyDGN engine
        self.torch_optimizer = None

        # to be set up after initialization of parameters
        self.current_depth = None

        self.n_obs = config["n_observations"]

        # store the quantile we want to use
        self.quantile = config["quantile"]

        # hidden layer size
        self.hidden_dim = config["hidden_dim"]

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
        if t_dist_cls == TruncatedDistribution:
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
            self.filter_network = FilterNetwork(dim_node_features, 1, config)
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
        self, data: Batch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        # first, determine if new layers have to be added
        self.update_depth()

        # computes probability vector of variational distr. q(layer)
        qL_probs = self.variational_L.compute_probability_vector()

        current_state = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # ---- COMPUTE MESSAGE FILTERING DISTRIBUTION --- #

        message_filters_list = []

        # next message will be remodulated when propagated to neighbors
        if self.filter_messages == "input_features":
            message_filters = self.filter_network(
                current_state
            )  # nodes x m(q)

            # bound to number of active layers
            message_filters = message_filters[:, : qL_probs.shape[0]]

            message_filters_list.append(message_filters)

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
            if self.filter_network is not None:
                if l > 0 and self.filter_messages == "embedding-no-weight-sharing":
                    message_filters = self.filter_network(
                        current_state, l - 1
                    )  # nodes x m(q)

                    # bound to number of active layers
                    message_filters = message_filters[:, : qL_probs.shape[0]]
                    # print(f'layer {l}')
                    # print(message_filters)
                    message_filters_list.append(message_filters)

            # ----------------------------------------------- #

            # NOTE: assuming source_to_target flow of messages
            if self.filter_network is not None:
                if l > 0 and self.filter_messages:
                    if self.filter_messages not in ["embedding-no-weight-sharing"]:
                        edge_msg_filter = message_filters[:, l][edge_index[0]]
                    else:
                        assert message_filters.shape[1] == 1, message_filters.shape
                        edge_msg_filter = message_filters[:, 0][edge_index[0]]
                else:
                    edge_msg_filter = None
            else:
                edge_msg_filter = None

            # computes log(p(theta)) for the hidden layers
            # assuming mu = 0 and a first order approximation
            if l > 0:
                # i-1 because we always have the first additional input layer
                hidden_layer = self.hidden_layers[l - 1]
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

            # FIXME This is what happens in UDN paper, does not make sense to me
            # if qL_probs[l] != 0:
            #     current_output = self._output_forward(
            #         output_layer,
            #         current_state,
            #         edge_index,
            #         edge_attr,
            #         data,
            #         l
            #     )
            # else:
            #     # Base case of first layer (at least according to original paper)
            #     # No weight on this layer, we compute just for inspection; no gradient to propagate
            #     current_output = self._output_forward(
            #         output_layer,
            #         current_state.detach(),
            #         edge_index,
            #         edge_attr,
            #         data,
            #         l
            #     )
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

        if self.filter_messages is not None:
            message_filters_list = torch.stack(message_filters_list, dim=1)
        else:
            message_filters_list = None

        return (
            output_state_list,  # num_nodes x layers x target_dimension
            hidden_state_list   # num_nodes x layers x dim_embedding
            if not self.return_fake_embeddings
            else torch.zeros(output_state_list.shape[0], 1),
            (
                data.batch,
                log_p_theta_hidden_list,
                log_p_theta_output_list,
                log_p_L_list,
                entropy_qL,
                qL_probs,  # 1 x layers   (output_state_list*(ql_probs.unsqueeze(2)) ).sum(1)
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