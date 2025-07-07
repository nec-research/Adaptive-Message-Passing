# Adaptive Message Passing (AMP)

This is the official code to reproduce the experiments of our ICML submission

[Adaptive Message Passing: A General Framework to Mitigate Oversmoothing, Oversquashing, and Underreaching](https://arxiv.org/abs/2312.16560)

**Authors:**
- Federico Errica
- Henrik Christiansen
- Viktor Zaverkin
- Takashi Maruyama
- Mathias Niepert
- Francesco Alesiani

## Citing us

    @inproceedings{errica_adaptive_2025,
      title={Adaptive Message Passing: A General Framework to Mitigate Oversmoothing, Oversquashing, and Underreaching},
      author={Errica, Federico and Christiansen, Henrik and Zaverkin, Viktor and Maruyama, Takashi and Niepert, Mathias and Alesiani, Francesco},
      booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
      year={2025}
    }

# Reproducing our experiments

Go to [this release](https://github.com/nec-research/Adaptive-Message-Passing/releases/tag/ICML-2025-Reproducibility) and follow the instructions to reproduce the results in our paper.

# Minimal Example Instructions

## Install a new environment with all requirements (Python 3.10)

    python3.10 -m venv ~/.venv/amp
    source ~/.venv/amp/bin/activate
    pip install torch>=2.6.0
    pip install torch-geometric>=2.6.0

## Files in this repository

The library is divided into multiple files:

- `distribution.py`: this file contains the distributions over the number of layers. You can choose between `Poisson`
   and `FoldedNormal` (see `example.py`). In principle you should not touch this file: if you think you need more expressive
   distributions, such as a mixture of Folded Normal distributions, please let us know.
- `layer_generator.py`: this file contains the code to generate layers of Graph Isomorphism Network (GIN) plus the edge
  filter implemented in AMP. This convolution is used directly in `model.py`. If you want to use a different convolution,
  you should implement a layer generator in this file and modify AMP (`model.py`) to accept layer generators as a parameter.
- `util.py` contains a few utility functions to transform a dotted string into a class
- `model.py` contains the implementation of AMP. You should not modify this file unless you need to train different deep
  graph networks like GCN, GAT, etc. If that is the case, you should parametrize AMP to accept new layer generators (
  see `layer_generator.py`)
- `example.py` contains a toy example that trains AMP-GIN on the ZINC regression dataset. We walk through the main choices
  to make in the next section

## Example.py walkthrough

This file is very straightforward. It is composed of
- Data loading
- **Hyper-parameter choices**
- Model instantiation
- Training Loop

and it is meant to show how the depth of AMP varies. 

Below we describe the meaning of the hyper-parameters:

    filter_type = "embeddings"  # or None or input_features

This variable describes the type of filtering you want to apply (None, node-inpuy-feature based, or node-embedding based). 
See our paper if you do not know what we are talking about.

    global_aggregation = True

This variable specifies if we are dealing with node (`False`) or graph (`True`) predictions.

    task = "regression_mae"  # OR regression_mse OR classification

This variable specifies the loss to apply. If regression, you can choose between MAE and MSE, otherwise you can choose
classification for multiclass classification tasks.

    hidden_dim = 64

This variable specifies the size of the node embeddings produced by the model.

    quantile = 0.99

This is the quantile at which we want to truncate our distribution over layers. It is recommended to leave it like this.

    learning_rate = 0.001

Learning rate of the Adam optimizer.

    layers_prior = Normal(loc=5.0, scale=10.0)

This variable specifies our prior assumptions about the **layers'** distribution. If the learned distribution deviates from this,
the model will be penalized. Think of it as a regularization. It can be set to `None`.

    theta_prior_scale = 10.0

This variable specifies our prior assumptions about the **parameters'** distribution. It encourages parameters to stay close to
a normal distribution with this standard deviation. It can be set to `None`.


    layer_variational_distribution = TruncatedDistribution(
        truncation_quantile=quantile,
        **{
            "discretized_distribution": {
                "class_name": "distribution.DiscretizedDistribution",
                "args": {"base_distribution":
                    {
                        "class_name": "distribution.FoldedNormal",
                        "args": {"loc": 5,
                                 "scale": 3},
                    }
                }
            }
        },
    )

This variable specifies the shape of the layers' distribution we want to use. In the case of a Folded Normal, we have
to discretize it first. Instead, if you want to use a Poisson, you can do something like this

    layer_variational_distribution = TruncatedDistribution(
        truncation_quantile=quantile,
        **{
            "discretized_distribution": {
                "class_name": "distribution.Poisson",
                "args": {"rate": 5},
            }
        },
    )

## Launch the example

The code will run on CUDA if it finds a GPU available (look for `device`). Also, batch size is fixed to 64 (look for `bs`)
but feel free to change it to suit your needs. Then you can launch the experiment with

    python example.py
