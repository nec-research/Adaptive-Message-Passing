#   Adaptive Message Passing
	
#   Authors: Federico Errica (Federico.Errica@neclab.eu) 
#            Henrik Christiansen (Henrik.Christiansen@neclab.eu)
# 	    Viktor Zaverkin (Viktor.Zaverkin@neclab.eu)
#   	    Takashi Maruyama (Takashi.Maruyama@neclab.eu)
#  	    Mathias Niepert (mathias.niepert@ki.uni-stuttgart.de)
#  	    Francesco Alesiani (Francesco.Alesiani @neclab.eu)
  
#   Files:    
#             distribution.py, 
#             layer_generator.py, 
#             model.py, 
#             util.py,
#             example.py 
            
# NEC Laboratories Europe GmbH, Copyright (c) 2025-, All rights reserved.  

#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
 
#        PROPRIETARY INFORMATION ---  

# SOFTWARE LICENSE AGREEMENT

# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.

# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor. 

# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).

# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.

# COPYRIGHT: The Software is owned by Licensor.  

# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.

# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.

# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.

# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.

# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.

# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.

# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.  

# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.

# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.

# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.

# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.  

# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.

# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.

# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.

# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.

# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.

# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.

import math
from typing import Tuple, Optional, List, Mapping, Any, Union

import torch
from torch import sigmoid
from torch.distributions import Distribution
from torch.nn import (
    ModuleList,
    Linear,
    Sequential,
    Tanh,
    Module,
    CrossEntropyLoss,
)
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_mean_pool, \
    global_max_pool

from distribution import (
    TruncatedDistribution,
    MixtureTruncated,
)
from layer_generator import GIN_LayerGenerator


class MessageFilter(Module):
    def __init__(self, input_features, hidden_dim, filter_type):
        super(MessageFilter, self).__init__()

        self.input_features = input_features
        self.hidden_dim = hidden_dim

        self.filter_type = filter_type
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


class AMP(Module):
    def __init__(
        self,
        dim_node_features: int,
        dim_target: int,
        size_training_set: int,
        filter_type: str,
        layer_variational_distribution: Union[TruncatedDistribution, MixtureTruncated],
        global_aggregation: bool,
        task: str,
        hidden_dim: int = 64,
        quantile: float = 0.99,
        layers_prior: Distribution = None,
        theta_prior: float = None,
        keep_all_layers: bool = False
    ):
        super().__init__()

        self.task = task

        # hidden layer size
        self.hidden_dim = hidden_dim

        # to be set up later by the PyDGN engine
        self.torch_optimizer = None

        # to be set up after initialization of parameters
        self.current_depth = None

        self.n_obs = size_training_set

        # store the quantile we want to use
        self.quantile = quantile

        # instantiate hidden and output layer generators
        self.layer_generator = GIN_LayerGenerator()
        (
            self.hidden_generator,
            self.output_generator,
        ) = self.layer_generator.make_generators(
            dim_node_features, self.hidden_dim, dim_target, global_aggregation
        )

        # these lists of layers will be dynamically updated
        self.hidden_layers = ModuleList([])

        # create the very first mapping from input to output
        self.output_layers = ModuleList([self.output_generator(layer_id=-1)])

        self.variational_L = layer_variational_distribution

        # # Instantiate the variational distribution q(\theta | \ell)
        # NOTE: not needed, see comment in forward method
        # q_theta_L_cls, q_theta_L_args = s2c(config['q_theta_given_L'])
        # q_theta_L = q_theta_L_cls(q_theta_L_args)
        # self.variational_theta = q_theta_L

        # prior scale for p(theta) - we use a normal with mean 0
        # If None, uninformative prior
        self.theta_prior_scale = theta_prior

        # prior mean and scale for p(ell). If None, uninformative prior
        self.layer_prior = layers_prior

        self.filter_messages = filter_type

        if self.filter_messages is not None:
            self.filter_network = MessageFilter(
                dim_node_features, hidden_dim, filter_type
            )
        else:  # no filtering
            self.filter_network = None

        self.keep_all_layers = keep_all_layers

        self.global_aggregation = global_aggregation

        self.device = None

    def to(self, device):
        """Set the device of the model."""
        super(AMP, self).to(device)

        self.device = device

        if self.filter_network:
            self.filter_network.to(device)
        self.variational_L.to(device)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        """
        Iteratively tries to match the right amount of layers with the checkpoint
        """
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

        print(f"Checkpoint loaded, {self.current_depth} layers")

    def update_depth(self, force_depth: int = -1):
        """
        Compute the current maximal depth of the variational
        posterior q(L) and create new layers if needed.

        Adapted from Nazaret and Blei's paper (ICML 2022)
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

            if self.filter_messages:
                self.filter_network.add_one_layer_to_filter(self.torch_optimizer)

        if self.current_depth < len(self.hidden_layers):
            if not self.keep_all_layers:
                # remove all extra layers but for an extra one
                # note: different from AMP paper but memory efficient
                self.hidden_layers = self.hidden_layers[:self.current_depth + 2]
                self.output_layers = self.output_layers[:self.current_depth + 2]

    def set_optimizer(self, optimizer):
        """
        Set the optimizer to later add the dynamically created
           layers' parameters to it.
        """
        self.torch_optimizer = optimizer

    def get_q_ell_named_parameters(self) -> dict:
        return self.variational_L.get_q_ell_named_parameters()

    def _hidden_forward(
        self,
        hidden_layer,
        current_state,
        edge_index,
        edge_msg_filter,
        layer_id,
    ):
        assert layer_id > 0

        if layer_id == 1:
            return hidden_layer(current_state)
        else:
            return hidden_layer(current_state, edge_index, edge_msg_filter)

    def _output_forward(
        self,
        output_layer,
        current_state,
        data,
    ):
        if self.global_aggregation:
            tmp_output = torch.cat(
                [
                    global_add_pool(current_state, data.batch),
                    global_max_pool(current_state, data.batch),
                    global_mean_pool(current_state, data.batch),
                ],
                dim=1,
            )
        else:
            tmp_output = current_state

        return output_layer(tmp_output)

    def forward(
        self, data: Batch, retain_grad=False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        # first, determine if new layers have to be added
        self.update_depth()

        x, edge_index, edge_attr, batch = (
            data.x.float(),
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

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

                # message_filters_list.append(message_filters)

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

                # i-1 because we always have the first additional input layer
                if retain_grad:
                    current_state.retain_grad()

                current_state = self._hidden_forward(
                    hidden_layer,
                    current_state,
                    edge_index,
                    edge_msg_filter,
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
                log_p_L = self.layer_prior.log_prob(torch.tensor([l - 1.0])).to(
                    self.device
                )
            else:
                log_p_L = torch.zeros(1).to(self.device)

            # compound hidden prior probs of each layer
            # due to the double summation
            log_theta_hidden_cumulative += log_p_theta_hidden
            log_theta_output_cumulative += log_p_theta_output

            current_output = self._output_forward(output_layer, current_state, data)

            # if l > 0:
            #     hidden_state_list.append(current_state)
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
            torch.distributions.Categorical(probs=qL_probs[1:]).entropy().unsqueeze(0)
        )  # shape [1]

        output_state_list = torch.stack(output_state_list, dim=1)  # ? x depth

        log_p_theta_hidden_list = torch.stack(
            log_p_theta_hidden_list, dim=1
        )  # 1 x depth

        log_p_theta_output_list = torch.stack(
            log_p_theta_output_list, dim=1
        )  # 1 x depth

        log_p_L_list = torch.stack(log_p_L_list, dim=1)  # 1 x depth

        qL_probs = qL_probs.unsqueeze(0)  # 1 x depth

        # if self.filter_network is not None:
        #     message_filters_list = torch.stack(message_filters_list, dim=1)
        # else:
        #     message_filters_list = None

        if self.task == "classification":
            elbo_fun = self._compute_classification
        elif self.task == "regression_mse":
            elbo_fun = self._compute_regression_mse
        elif self.task == "regression_mae":
            elbo_fun = self._compute_regression_mae

        elbo = elbo_fun(
            output_state_list,
            data.y,
            log_p_theta_hidden_list,
            log_p_theta_output_list,
            log_p_L_list,
            qL_probs,
            entropy_qL,
        )

        # Weight predictions at each layer
        # pred = (output_state * qL_log_probs.exp().unsqueeze(2)).sum(1)
        pred = (output_state_list * qL_probs.unsqueeze(2)).sum(1)

        return (
            pred,
            elbo,
        )

    def _compute_classification(
        self,
        output_state,
        targets,
        log_p_theta_hidden,
        log_p_theta_output,
        log_p_L,
        qL_probs,
        entropy_qL,
    ):
        # maximize log p y given x
        log_p_y_list = [
            -CrossEntropyLoss(reduction="mean")(
                output_state[:, i, :], targets
            ).unsqueeze(0)
            * self.n_obs
            for i in range(output_state.shape[1])
        ]

        log_p_y = torch.stack(log_p_y_list, dim=1)

        # (weighted) mean over layers
        elbo = log_p_y
        elbo += log_p_theta_hidden
        elbo += log_p_theta_output
        elbo += log_p_L

        elbo *= qL_probs

        elbo = elbo.sum(1)
        elbo += entropy_qL
        elbo = elbo / self.n_obs

        return -elbo.mean(0)

    def _compute_regression_mse(
        self,
        output_state,
        targets,
        log_p_theta_hidden,
        log_p_theta_output,
        log_p_L,
        qL_probs,
        entropy_qL,
    ):
        two = torch.Tensor([2.0]).to(targets.device)
        pi = torch.Tensor([math.pi]).to(targets.device)

        if len(targets.shape) == 1:
            targets = targets.unsqueeze(1)

        if len(output_state.shape) == 2:
            output_state = output_state.unsqueeze(2)

        # maximize log p y given x (gaussian nll with variance 1)
        log_p_y_list = [
            (-torch.mean(((output_state[:, i, :] - targets) ** 2).sum(1)) / two)
            * self.n_obs  # - torch.log(two * pi) / two
            for i in range(output_state.shape[1])
        ]

        log_p_y = torch.stack(log_p_y_list, dim=1)

        # (weighted) mean over layers
        elbo = log_p_y
        elbo += log_p_theta_hidden
        elbo += log_p_theta_output
        elbo += log_p_L

        elbo *= qL_probs

        elbo = elbo.sum(1)
        elbo += entropy_qL
        elbo = elbo / self.n_obs

        return -elbo.mean(0)

    def _compute_regression_mae(
        self,
        output_state,
        targets,
        log_p_theta_hidden,
        log_p_theta_output,
        log_p_L,
        qL_probs,
        entropy_qL,
    ):
        if len(targets.shape) == 1:
            targets = targets.unsqueeze(1)

        if len(output_state.shape) == 2:
            output_state = output_state.unsqueeze(2)

        one = torch.Tensor([1.0]).to(targets.device)

        # maximize -MAE
        log_p_y_list = [
            (-torch.mean((torch.abs(output_state[:, i, :] - targets)).sum(1)) / one)
            * self.n_obs
            for i in range(output_state.shape[1])
        ]

        log_p_y = torch.stack(log_p_y_list, dim=1)

        # (weighted) mean over layers
        elbo = log_p_y
        elbo += log_p_theta_hidden
        elbo += log_p_theta_output
        elbo += log_p_L

        elbo *= qL_probs

        elbo = elbo.sum(1)
        elbo += entropy_qL
        elbo = elbo / self.n_obs
        return -elbo.mean(0)
