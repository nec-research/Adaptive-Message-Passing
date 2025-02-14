from pydgn.evaluation.config import Config
from pydgn.experiment.supervised_task import SupervisedTask
from pydgn.training.engine import TrainingEngine

from model import UnboundedDepthNetwork


class UDN_SupervisedExperiment(SupervisedTask):
    def _create_engine(
        self,
        config: Config,
        model: UnboundedDepthNetwork,
        device: str,
        evaluate_every: int,
        reset_eval_model_hidden_state: bool,
    ) -> TrainingEngine:
        """
        Sets the optimizer into the model to allow for dynamic optimization
        of newly inserted layers.
        """
        engine = super(UDN_SupervisedExperiment, self)._create_engine(
            config,
            model,
            device,
            evaluate_every,
            reset_eval_model_hidden_state,
        )
        model.set_optimizer(engine.optimizer)
        return engine
