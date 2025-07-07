import math
from typing import List, Tuple

import numpy as np
import torch
from pydgn.training.callback.metric import Metric, MulticlassAccuracy, \
    MeanSquareError, MeanAverageError, Classification, MulticlassClassification
from sklearn.metrics import average_precision_score
from torch import softmax
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_scatter import scatter


class UDN_ELBO_Classification(Metric):
    @property
    def name(self) -> str:
        return "UDN_ELBO_Classification"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            output_state,
            hidden_state,
            (
                batch,
                log_p_theta_hidden,
                log_p_theta_output,
                log_p_L,
                entropy_qL,
                qL_probs,
                n_obs,
                message_filters
            ),
        ) = outputs

        # maximize log p y given x
        log_p_y_list = [
            -CrossEntropyLoss(reduction="mean")(
                output_state[:, i, :], targets
            ).unsqueeze(0)
            * n_obs
            for i in range(output_state.shape[1])
        ]

        log_p_y = torch.stack(log_p_y_list, dim=1)

        # (weighted) mean over layers
        elbo = log_p_y
        elbo += log_p_theta_hidden
        elbo += log_p_theta_output
        elbo += log_p_L

        # FIXME This is what happens in UDN paper, qL_probs[0] = 0. but just for convenience
        # elbo *= torch.cat((torch.ones(1,1).to(elbo.device), qL_probs[:, 1:]), dim=1)

        # this is what makes most sense (that is, learn prob for first layer as well)
        elbo *= qL_probs

        elbo = elbo.sum(1)
        elbo += entropy_qL
        elbo = elbo / n_obs

        return elbo, targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        elbo = predictions
        # to maximize the elbo we need to minimize -elbo
        return -elbo.mean(0)  # sum over samples


class UDN_ELBO_Regression(Metric):
    @property
    def name(self) -> str:
        return "UDN_ELBO_Regression"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            output_state,
            hidden_state,
            (
                batch,
                log_p_theta_hidden,
                log_p_theta_output,
                log_p_L,
                entropy_qL,
                qL_probs,
                n_obs,
                message_filters
            ),
        ) = outputs

        two = torch.Tensor([2.0]).to(targets.device)
        pi = torch.Tensor([math.pi]).to(targets.device)

        if len(targets.shape) == 1:
            targets = targets.unsqueeze(1)

        if len(output_state.shape) == 2:
            output_state = output_state.unsqueeze(2)

        # maximize log p y given x (gaussian nll with variance 1)
        log_p_y_list = [
            (-torch.mean(((output_state[:, i, :] - targets) ** 2).sum(1)) / two)
            * n_obs #- torch.log(two * pi) / two
            for i in range(output_state.shape[1])
        ]

        log_p_y = torch.stack(log_p_y_list, dim=1)

        # (weighted) mean over layers
        elbo = log_p_y
        elbo += log_p_theta_hidden
        elbo += log_p_theta_output
        elbo += log_p_L

        # FIXME This is what happens in UDN paper, qL_probs[0] = 0. but just for convenience
        # elbo *= torch.cat((torch.ones(1,1).to(elbo.device), qL_probs[:, 1:]), dim=1)

        # this is what makes most sense (that is, learn prob for first layer as well)
        elbo *= qL_probs

        elbo = elbo.sum(1)
        elbo += entropy_qL
        elbo = elbo / n_obs
        return elbo, targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        elbo = predictions
        # to maximize the elbo we need to minimize -elbo
        return -elbo.mean(0)  # sum over samples


class UDN_ELBO_RegressionMAE(Metric):
    @property
    def name(self) -> str:
        return "UDN_ELBO_Regression MAE"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            output_state,
            hidden_state,
            (
                batch,
                log_p_theta_hidden,
                log_p_theta_output,
                log_p_L,
                entropy_qL,
                qL_probs,
                n_obs,
                message_filters
            ),
        ) = outputs

        if len(targets.shape) == 1:
            targets = targets.unsqueeze(1)

        if len(output_state.shape) == 2:
            output_state = output_state.unsqueeze(2)

        one = torch.Tensor([1.0]).to(targets.device)

        # maximize -MAE
        log_p_y_list = [
            (-torch.mean((torch.abs(output_state[:, i, :] - targets)).sum(1)) / one)
            * n_obs
            for i in range(output_state.shape[1])
        ]

        log_p_y = torch.stack(log_p_y_list, dim=1)

        # (weighted) mean over layers
        elbo = log_p_y
        elbo += log_p_theta_hidden
        elbo += log_p_theta_output
        elbo += log_p_L

        # FIXME This is what happens in UDN paper, qL_probs[0] = 0. but just for convenience
        # elbo *= torch.cat((torch.ones(1,1).to(elbo.device), qL_probs[:, 1:]), dim=1)

        # this is what makes most sense (that is, learn prob for first layer as well)
        elbo *= qL_probs

        elbo = elbo.sum(1)
        elbo += entropy_qL
        elbo = elbo / n_obs
        return elbo, targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        elbo = predictions
        # to maximize the elbo we need to minimize -elbo
        return -elbo.mean(0)  # sum over samples


class UDN_WeightedLogLikelihoodClassification(Metric):
    @property
    def name(self) -> str:
        return "UDN Weighted Log-Likelihood Classification"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            output_state,
            hidden_state,
            (
                batch,
                log_p_theta_hidden,
                log_p_theta_output,
                log_p_L,
                entropy_qL,
                qL_probs,
                n_obs,
                message_filters
            ),
        ) = outputs

        # maximize log p y given x
        log_p_y_list = [
            -CrossEntropyLoss(reduction="sum")(
                output_state[:, i, :], targets
            ).unsqueeze(0)
            for i in range(output_state.shape[1])
        ]

        # qL_probs = qL_log_probs.exp()

        log_p_y = torch.stack(log_p_y_list, dim=1)
        log_p_y *= qL_probs
        log_p_y = log_p_y.sum(1)  # (weighted) sum over layers
        return log_p_y, targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        # to maximize the elbo we need to minimize -elbo
        return predictions.mean(0)  # sum over samples


class UDN_WeightedLogLikelihoodRegression(Metric):
    @property
    def name(self) -> str:
        return "UDN Weighted Log-Likelihood Regression"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            output_state,
            hidden_state,
            (
                batch,
                log_p_theta_hidden,
                log_p_theta_output,
                log_p_L,
                entropy_qL,
                qL_probs,
                n_obs,
                message_filters
            ),
        ) = outputs

        two = torch.Tensor([2.0]).to(targets.device)
        pi = torch.Tensor([math.pi]).to(targets.device)

        if len(targets.shape) == 1:
            targets = targets.unsqueeze(1)

        if len(output_state.shape) == 2:
            output_state = output_state.unsqueeze(2)

        # maximize log p y given x (gaussian nll with variance 1)
        log_p_y_list = [
            (-torch.mean(((output_state[:, i, :] - targets) ** 2).sum(1)) / two)
            * n_obs #- torch.log(two * pi) / two
            for i in range(output_state.shape[1])
        ]

        # qL_probs = qL_log_probs.exp()

        log_p_y = torch.stack(log_p_y_list, dim=1)
        log_p_y *= qL_probs
        log_p_y = log_p_y.sum(1)  # (weighted) sum over layers
        return log_p_y, targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        # to maximize the elbo we need to minimize -elbo
        return predictions.mean(0)  # sum over samples


class UDN_theta_hidden(Metric):
    @property
    def name(self) -> str:
        return "UDN_theta_hidden"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            output_state,
            hidden_state,
            (
                batch,
                log_p_theta_hidden,
                log_p_theta_output,
                log_p_L,
                entropy_qL,
                qL_probs,
                n_obs,
                message_filters
            ),
        ) = outputs
        return log_p_theta_hidden.sum(1), targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        # to maximize the elbo we need to minimize -elbo
        return predictions.mean(0)  # sum over samples


class UDN_entropy(Metric):
    @property
    def name(self) -> str:
        return "UDN_entropy"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            output_state,
            hidden_state,
            (
                batch,
                log_p_theta_hidden,
                log_p_theta_output,
                log_p_L,
                entropy_qL,
                qL_probs,
                n_obs,
                message_filters
            ),
        ) = outputs
        return entropy_qL.unsqueeze(1), targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        # to maximize the elbo we need to minimize -elbo
        return predictions.mean(0)  # sum over samples


class UDN_prior_layer(Metric):
    @property
    def name(self) -> str:
        return "UDN_prior_layer"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            output_state,
            hidden_state,
            (
                batch,
                log_p_theta_hidden,
                log_p_theta_output,
                log_p_L,
                entropy_qL,
                qL_probs,
                n_obs,
                message_filters
            ),
        ) = outputs
        return log_p_L.sum(1), targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        # to maximize the elbo we need to minimize -elbo
        return predictions.mean(0)  # sum over samples


class UDN_theta_output(Metric):
    @property
    def name(self) -> str:
        return "UDN_theta_output"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            output_state,
            hidden_state,
            (
                batch,
                log_p_theta_hidden,
                log_p_theta_output,
                log_p_L,
                entropy_qL,
                qL_probs,
                n_obs,
                message_filters
            ),
        ) = outputs
        return log_p_theta_output.mean(1), targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        # to maximize the elbo we need to minimize -elbo
        return predictions.mean(0)  # sum over samples


class UDN_Accuracy(MulticlassAccuracy):
    @property
    def name(self) -> str:
        return "UDN_Accuracy"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            output_state,
            hidden_state,
            (
                batch,
                log_p_theta_hidden,
                log_p_theta_output,
                log_p_L,
                entropy_qL,
                qL_probs,
                n_obs,
                message_filters
            ),
        ) = outputs

        # Weight predictions at each layer
        # pred = (output_state * qL_log_probs.exp().unsqueeze(2)).sum(1)
        pred = (output_state * qL_probs.unsqueeze(2)).sum(1)

        correct = self._get_correct(pred)

        if len(targets.shape) == 2:
            targets = targets.squeeze(dim=1)

        return correct, targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        metric = (
            100.0 * (predictions == targets).sum().float() / targets.size(0)
        )
        return metric


class UDN_Depth(Metric):
    @property
    def name(self) -> str:
        return "UDN_Depth"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            output_state,
            hidden_state,
            (
                batch,
                log_p_theta_hidden,
                log_p_theta_output,
                log_p_L,
                entropy_qL,
                qL_probs,
                n_obs,
                message_filters,
            ),
        ) = outputs

        return torch.tensor([qL_probs.shape[1]]).unsqueeze(0), targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        return float(predictions.max())


# IMPLEMENT MSE for Synthetic tasks the same way of ADGN paper
# THIS IS FOR NODE PREDICTION TASKS
class GraphWiseMSE(Metric):

    @property
    def name(self) -> str:
        """
        The name of the loss to be used in configuration files and displayed
        on Tensorboard
        """
        return "Graph-wise Mean Square Error"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        preds, batch = outputs[0], outputs[2][0]

        if len(preds.shape) == 3:  # num_nodes/graphs x layers x output size
            # UDN CASE, compute weighted prediction over layers
            qL_probs = outputs[2][5]
            preds *= qL_probs.unsqueeze(2)
            preds = preds.sum(1)

        nodes_in_graph = scatter(torch.ones(batch.shape[0]).to(preds.device),
                                 batch).unsqueeze(1).to(preds.device)
        # nodes_in_graph = torch.tensor([[(batch == i).sum()] for i in range(max(batch)+1)]).to(device)
        nodes_loss = (preds - targets.reshape(targets.shape[0], 1)) ** 2

        # Implementing global add pool of the node losses, reference here
        # https://github.com/cvignac/SMP/blob/62161485150f4544ba1255c4fcd39398fe2ca18d/multi_task_utils/util.py#L99
        graphs_loss = global_add_pool(nodes_loss,
                                batch) / nodes_in_graph  # average_nodes

        # nodes_loss = (preds.reshape(preds.shape[0], 1) - targets.reshape(targets.shape[0], 1)) ** 2
        # graphs_loss = global_mean_pool(nodes_loss, batch)

        return graphs_loss, targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        return predictions.mean(0)


# IMPLEMENT LOG 10 MSE for Synthetic tasks the same way of ADGN paper
class GraphWiseLogMSE(GraphWiseMSE):

    @property
    def name(self) -> str:
        """
        The name of the loss to be used in configuration files and displayed
        on Tensorboard
        """
        return "Log 10 Graph-wise Mean Square Error"

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        return torch.log10(super().compute_metric(targets, predictions))


# IMPLEMENT MSE for Synthetic tasks the same way of ADGN paper
# THIS IS FOR GRAPH PREDICTION TASKS
class GraphPredMSE(Metric):

    @property
    def name(self) -> str:
        """
        The name of the loss to be used in configuration files and displayed
        on Tensorboard
        """
        return "Mean Square Error"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        preds, batch = outputs[0], outputs[2][0]

        if len(preds.shape) == 3:  # num_nodes/graphs x layers x output size
            # UDN CASE, compute weighted prediction over layers
            qL_probs = outputs[2][5]
            preds *= qL_probs.unsqueeze(2)
            preds = preds.sum(1)

        # nodes_in_graph = torch.tensor([[(batch == i).sum()] for i in range(max(batch)+1)]).to(device)
        graphs_loss = (preds - targets.reshape(targets.shape[0], 1)) ** 2

        return graphs_loss, targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        return predictions.mean(0)


# IMPLEMENT LOG 10 MSE for Synthetic tasks the same way of ADGN paper
class GraphPredLogMSE(GraphPredMSE):

    @property
    def name(self) -> str:
        """
        The name of the loss to be used in configuration files and displayed
        on Tensorboard
        """
        return "Log 10 Mean Square Error"

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        return torch.log10(super().compute_metric(targets, predictions))


class UDN_MAE(MeanAverageError):

    @property
    def name(self) -> str:
        """
        The name of the loss to be used in configuration files and displayed
        on Tensorboard
        """
        return "Mean Average Error"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        preds, batch = outputs[0], outputs[2][0]

        if len(preds.shape) == 3:  # num_nodes/graphs x layers x output size
            # UDN CASE, compute weighted prediction over layers
            qL_probs = outputs[2][5]
            preds *= qL_probs.unsqueeze(2)
            preds = preds.sum(1)

        if len(targets.shape) == 1:
            targets = targets.unsqueeze(1)

        return preds, targets


class MultiLabelCrossEntropy(Classification):
    # https://github.com/toenshoff/LRGB/blob/main/graphgps/loss/multilabel_classification_loss.py

    @property
    def name(self) -> str:
        return 'Multilabel Cross Entropy Loss'

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = outputs[0]
        assert len(targets.shape) == 2
        return outputs, targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:

        bce_loss = BCEWithLogitsLoss()
        is_labeled = targets == targets  # Filter our nans.

        return bce_loss(predictions[is_labeled], targets[is_labeled].float())

class OGB_AP(Metric):
    # https://github.com/toenshoff/LRGB/blob/main/graphgps/metrics_ogb.py

    @property
    def name(self) -> str:
        return 'OGB Average Precision'


    def get_predictions_and_targets(
            self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = outputs[0]
        assert len(targets.shape) == 2
        return outputs, targets


    def compute_metric(
            self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:

        ap_list = []

        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        for i in range(targets.shape[1]):

            # AUC is only defined when there is at least one positive data.
            if np.sum(targets[:, i] == 1) > 0 and np.sum(targets[:, i] == 0) > 0:
                # ignore nan values
                is_labeled = targets[:, i] == targets[:, i]
                ap = average_precision_score(targets[is_labeled, i],
                                             predictions[is_labeled, i])

                ap_list.append(ap)

        if len(ap_list) == 0:
            raise RuntimeError(
                'No positively labeled data available. Cannot compute Average Precision.')

        return torch.tensor([sum(ap_list) / len(ap_list)])


class UDN_ELBO_MultiLabelClassification(Metric):
    @property
    def name(self) -> str:
        return "UDN_ELBO_MultiLabelClassification"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            output_state,
            hidden_state,
            (
                batch,
                log_p_theta_hidden,
                log_p_theta_output,
                log_p_L,
                entropy_qL,
                qL_probs,
                n_obs,
                message_filters
            ),
        ) = outputs

        # maximize log p y given x
        bce_loss = BCEWithLogitsLoss()
        is_labeled = targets == targets  # Filter out nans.

        assert len(targets.shape) == 2

        targets = targets[is_labeled].float()

        log_p_y_list = [
            -bce_loss(
                output_state[:, i, :][is_labeled], targets
            ).unsqueeze(0)
            * n_obs
            for i in range(output_state.shape[1])
        ]

        log_p_y = torch.stack(log_p_y_list, dim=1)


        # (weighted) mean over layers
        elbo = log_p_y
        elbo += log_p_theta_hidden
        elbo += log_p_theta_output
        elbo += log_p_L

        # FIXME This is what happens in UDN paper, qL_probs[0] = 0. but just for convenience
        # elbo *= torch.cat((torch.ones(1,1).to(elbo.device), qL_probs[:, 1:]), dim=1)

        # this is what makes most sense (that is, learn prob for first layer as well)
        elbo *= qL_probs

        elbo = elbo.sum(1)
        elbo += entropy_qL
        elbo = elbo / n_obs
        return elbo, targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        elbo = predictions
        # to maximize the elbo we need to minimize -elbo
        return -elbo.mean(0)  # sum over samples


class UDN_OGB_AP(OGB_AP):

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            output_state,
            hidden_state,
            (
                batch,
                log_p_theta_hidden,
                log_p_theta_output,
                log_p_L,
                entropy_qL,
                qL_probs,
                n_obs,
                message_filters
            ),
        ) = outputs

        # Weight predictions at each layer
        # pred = (output_state * qL_log_probs.exp().unsqueeze(2)).sum(1)
        pred = (output_state * qL_probs.unsqueeze(2)).sum(1)

        assert len(targets.shape) == 2

        return pred, targets


class SingleGraphClassification(MulticlassClassification):

    @property
    def name(self) -> str:
        return 'Single Graph Classification'

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        targets = outputs[2][1]
        preds = outputs[0]
        return preds, targets


class SingleGraphAccuracy(MulticlassAccuracy):

    @property
    def name(self) -> str:
        return 'Single Graph Accuracy'

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = outputs[0]
        targets = outputs[2][1]
        correct = self._get_correct(preds)
        return correct, targets




class UDN_ELBO_SingleGraphClassification(MulticlassClassification):

    @property
    def name(self) -> str:
        return 'Single Graph Classification'

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            output_state,
            hidden_state,
            (
                batch,
                log_p_theta_hidden,
                log_p_theta_output,
                log_p_L,
                entropy_qL,
                qL_probs,
                n_obs,
                message_filters,
                eval_indices
            ),
        ) = outputs

        # maximize log p y given x
        log_p_y_list = [
            -CrossEntropyLoss(reduction="mean")(
                output_state[eval_indices, i, :], targets[eval_indices]
            ).unsqueeze(0)
            * n_obs
            for i in range(output_state.shape[1])
        ]

        log_p_y = torch.stack(log_p_y_list, dim=1)

        # (weighted) mean over layers
        elbo = log_p_y
        elbo += log_p_theta_hidden
        elbo += log_p_theta_output
        elbo += log_p_L

        # FIXME This is what happens in UDN paper, qL_probs[0] = 0. but just for convenience
        # elbo *= torch.cat((torch.ones(1,1).to(elbo.device), qL_probs[:, 1:]), dim=1)

        # this is what makes most sense (that is, learn prob for first layer as well)
        elbo *= qL_probs

        elbo = elbo.sum(1)
        elbo += entropy_qL
        elbo = elbo / n_obs

        return elbo, targets

    def compute_metric(
            self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        elbo = predictions
        # to maximize the elbo we need to minimize -elbo
        return -elbo.mean(0)  # sum over samples


class AMPSingleGraphAccuracy(MulticlassAccuracy):

    @property
    def name(self) -> str:
        return 'Single Graph Accuracy'

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        (
            output_state,
            hidden_state,
            (
                batch,
                log_p_theta_hidden,
                log_p_theta_output,
                log_p_L,
                entropy_qL,
                qL_probs,
                n_obs,
                message_filters,
                eval_indices
            ),
        ) = outputs

        # Weight predictions at each layer
        # pred = (output_state * qL_log_probs.exp().unsqueeze(2)).sum(1)
        pred = (output_state[eval_indices] * qL_probs.unsqueeze(2)).sum(1)

        correct = self._get_correct(pred)

        if len(targets.shape) == 2:
            targets = targets.squeeze(dim=1)

        return correct, targets[eval_indices]


class UDN_SingleGraphDepth(Metric):
    @property
    def name(self) -> str:
        return "UDN_Depth"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            output_state,
            hidden_state,
            (
                batch,
                log_p_theta_hidden,
                log_p_theta_output,
                log_p_L,
                entropy_qL,
                qL_probs,
                n_obs,
                message_filters,
                _
            ),
        ) = outputs

        return torch.tensor([qL_probs.shape[1]]).unsqueeze(0), targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        return float(predictions.max())