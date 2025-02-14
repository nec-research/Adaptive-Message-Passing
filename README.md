# Adaptive Message Passing (AMP)

This is the official code to reproduce the experiments of our JMLR submission

[Adaptive Message Passing: A General Framework to Mitigate Oversmoothing, Oversquashing, and Underreaching](https://arxiv.org/abs/2312.16560)

**Authors:**
- Federico Errica
- Henrik Christiansen
- Viktor Zaverkin
- Takashi Maruyama
- Mathias Niepert
- Francesco Alesiani

## Citing us

    @article{errica2023adaptive,
      title={Adaptive Message Passing: A General Framework to Mitigate Oversmoothing, Oversquashing, and Underreaching},
      author={Errica, Federico and Christiansen, Henrik and Zaverkin, Viktor and Maruyama, Takashi and Niepert, Mathias and Alesiani, Francesco},
      journal={arXiv preprint arXiv:2312.16560},
      year={2023}
    }

## How to reproduce our experiments
### Install all requirements (Python 3.10)
Create a new virtual environment with Python 3.10 installed. Then

    # install torch 2.0.1 among other dependencies
    pip install -r requirements.txt

    # install specific version of PyG to ensure no changes can influence results
    pip install torch_geometric==2.3.1

    # install ther dependencies (**CHANGE cu117 with your CUDA version**):
    pip install torch_scatter==2.1.1 torch_sparse==0.6.17 -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

### Data Preparation

Run the notebook `Data_Converter.ipynb` to prepare the datasets for the experiments.

### Launch experiments (assumes CUDA is used)

    source launch_synthetic_300patience.sh
    source launch_peptides.sh

Results will be stored in the `RESULTS` folder according to the [PyDGN](https://pydgn.readthedocs.io/en/latest/) format.

### Troubleshooting

Please open an issue if you have trouble running the experiments.

### Running on CPU

If you need to run on CPU, a few minor changes need to be made in the configuration files of the experiments.
In particular, use

    # Hardware
    device:  cuda
    max_cpus:  [NUM CPUs you want to use]
    max_gpus: 0
    gpus_per_task:  0
    
    # Data Loading
    dataset_getter: pydgn.data.provider.DataProvider
    data_loader:
      class_name: torch_geometric.loader.DataLoader
      args:
        num_workers : 0
        pin_memory: False

Please refer to [PyDGN](https://pydgn.readthedocs.io/en/latest/)'s documentation. 
