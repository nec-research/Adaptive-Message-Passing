import json
import os
from io import BytesIO

import PIL
import numpy as np
import torch
from matplotlib import pyplot as plt
from pydgn.training.callback.plotter import Plotter
from pydgn.training.event.state import State
from torch.distributions import Categorical


class UDNPlotter(Plotter):
    def on_epoch_end(self, state: State):
        super(UDNPlotter, self).on_epoch_end(state)

        qL_probs = state.model.variational_L.compute_probability_vector()

        plt.figure()
        plt.bar(np.arange(qL_probs.shape[0]), qL_probs.detach().cpu().numpy())
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close()

        # Convert image buffer to CHW tensor
        image = PIL.Image.open(buffer)
        image = image.convert("RGB")  # Convert to RGB format if necessary
        image = np.array(image)  # Convert PIL image to NumPy array
        image = np.transpose(image, (2, 0, 1))  # Convert HWC to CHW
        image_tensor = torch.ByteTensor(image)

        self.writer.add_image(
            tag="q(ell)", img_tensor=image_tensor, global_step=state.epoch
        )

        # self.writer.add_histogram(tag='q(ell)',
        #                           values=c.sample((1000,)),
        #                           global_step=state.epoch)


class StatsPlotter(UDNPlotter):

    def on_fit_start(self, state: State):
        super().on_fit_start(state)

        self.distribution_params = {}

        # update params file with new parameters for later
        params_filepath = os.path.join(self.exp_path, "parameters.torch")

        if os.path.exists(params_filepath):
            try:
                self.distribution_params = torch.load(params_filepath)
            except Exception as e:
                print(e)
                self.distribution_params = {}

    def on_epoch_end(self, state: State):
        super().on_epoch_end(state)

        named_parameters = state.model.get_q_ell_named_parameters()

        # detach and bring to cpu
        for k, v in named_parameters.items():
            named_parameters[k] = v.detach().cpu()

        # print on tensorboard
        for k, v in named_parameters.items():
            assert len(v.shape) == 1

            if v.shape[0] == 1:  # scalars
                self.writer.add_scalar(
                    tag=k, scalar_value=v.item(), global_step=state.epoch
                )
            elif v.shape[0] > 1:
                for i in range(v.shape[0]):
                    self.writer.add_scalar(
                        tag=f'{k}_{i}',
                        scalar_value=v[i].item(),
                        global_step=state.epoch
                    )
                # self.writer.add_histogram(
                #     tag=k,
                #     values=v,
                #     global_step=state.epoch,
                #     bins="auto",
                # )

        self.distribution_params[f"{int(state.epoch)}"] = named_parameters

        # update params file with new parameters for later
        try:
            params_filepath = os.path.join(self.exp_path, "parameters.torch")
            torch.save(self.distribution_params, params_filepath)
        except Exception as e:
            print(e)

    def on_fit_end(self, state: State):
        super().on_fit_end(state)

        # update params file with new parameters for later
        params_filepath = os.path.join(self.exp_path, "parameters.torch")
        try:
            torch.save(self.distribution_params, params_filepath)
        except Exception as e:
            print(e)