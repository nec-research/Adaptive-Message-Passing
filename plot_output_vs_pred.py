import json
import os.path as osp
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import *

import pydgn
from pydgn.data.dataset import TUDatasetInterface, DatasetInterface
from pydgn.experiment.util import s2c  # string to class

from pydgn.data.provider import DataProvider
from torch_geometric.loader import DataLoader

import sys

from matplotlib import pyplot as plt

import seaborn as sns
# Set Seaborn color palette to colorblind-friendly
sns.set_palette("colorblind")

from tueplots import figsizes, bundles
plt.rcParams.update(bundles.icml2024())

# Increase the resolution of all the plots below
plt.rcParams.update({"figure.dpi": 150})

plt.rcParams.update(figsizes.icml2024_half())

sys.path.append("/home/ferrica/PycharmProjects/adaptive-message-passing")
import dgn, adgn#, dataset

def get_model(udn, prop, network, folder):
    if udn:
        results_folder = f'{folder}/udn_{network}_{prop}'
    else:
        results_folder = f'{folder}/{network}_{prop}'
    dataset_root = 'DATA'
    dataset_name = prop
    if prop in ["Eccentricity","SSSP","Diameter"]:
        dataset_class = 'dataset.GraphPropertyPrediction'
    elif prop in ["peptides-func","peptides-struct"]:
        dataset_class = 'dataset.Peptides'

    device = 'cpu'

    config_file = osp.join(results_folder, 
                            'MODEL_ASSESSMENT',
                            'OUTER_FOLD_1',
                            'MODEL_SELECTION',
                            'winner_config.json')

    best_config = json.load(open(config_file, 'r'))['config']['supervised_config']
    print(best_config)

    best_score = 1e300
    best_folder = ""
    mean_score = 0
    for run_index in range(1,21):
        run_results_folder = osp.join(results_folder, 
                                'MODEL_ASSESSMENT',
                                'OUTER_FOLD_1',
                                f'final_run{run_index}')
        run_results_file = osp.join(run_results_folder,
                                f'run_{run_index}_results.torch')

        data = torch.load(run_results_file)
        score = data[2]["score"]["main_score"]#!!!
        #print(data)
        if score < best_score:
            best_folder = run_results_folder
            best_score = score
        mean_score += score
    print("Mean",mean_score/len(range(1,21)))
    print("Best", best_score)

    final_model_ckpt = osp.join(best_folder,
                        'best_checkpoint.pth')

    best_ckpt = torch.load(final_model_ckpt,map_location=torch.device('cpu'))['model_state']

    dataset = s2c(dataset_class)(dataset_root, dataset_name)
    dim_node_features = dataset.dim_node_features
    dim_edge_features = dataset.dim_edge_features
    num_classes = dataset.dim_target

    print(best_config['model'])
    #if udn:
    #    model = s2c("model.AMP")(dim_node_features, dim_edge_features, num_classes, None, best_config)
    #else:
    model = s2c(best_config['model'])(dim_node_features, dim_edge_features, num_classes, None, best_config)
    model.to(device)
    model.eval()
    model.load_state_dict(best_ckpt)
    return model

def get_test_data_loader(prop):
    batch_size = 32
    shuffle = False  # MUST STAY FALSE!

    dataset_root = 'DATA'
    dataset_name = prop
    if prop in ["Eccentricity","SSSP","Diameter"]:
        dataset_class = 'dataset.GraphPropertyPrediction'
    elif prop in ["peptides-func","peptides-struct"]:
        dataset_class = 'dataset.Peptides'
    data_loader_class = 'torch_geometric.loader.DataLoader'
    device = 'cpu'
    splits_filepath = f'DATA_SPLITS/{dataset_name}/{dataset_name}_outer1_inner1.splits'

    dataset_getter = DataProvider(dataset_root, splits_filepath, 
                              s2c(dataset_class), dataset_name,
                              s2c(data_loader_class), {},
                              outer_folds=1, inner_folds=1)
    dataset_getter.set_exp_seed(42)
    dataset_getter.set_inner_k(0)
    dataset_getter.set_outer_k(0)

    data_loader_args = dict(batch_size=batch_size, shuffle=shuffle)

    train_loader = dataset_getter.get_outer_train(**data_loader_args)
    val_loader = dataset_getter.get_outer_val(**data_loader_args)
    test_loader = dataset_getter.get_outer_test(**data_loader_args)
    return test_loader

def run_model(model, data_loader, use_mean_layer=False):
    predictions, targets, all_outputs = [], [], []
    with torch.no_grad():
        for batch in data_loader:
            batch.to('cpu')

            outputs = model(batch)
            all_outputs.append(outputs)

            if use_mean_layer:
                mean_layer = 0
                for l,weight in enumerate(outputs[2][5][0]):
                    mean_layer += l*weight
                mean_layer = round(mean_layer.item())

            preds, data = outputs[0], outputs[2][0]
            if len(preds.shape) == 3:
                for idx, target in enumerate(batch.y):
                    if use_mean_layer:
                        this_predicition = outputs[0][idx][mean_layer]
                    else:
                        this_predicition = 0
                        for idx2, pred in enumerate(preds[idx]):
                            this_predicition += outputs[0][idx][idx2] * outputs[2][5][0][idx2]
                    predictions.append(this_predicition)
                targets.append(batch.y.detach().cpu())   
            else:
                predictions.append(outputs[0].squeeze(1))
                targets.append(batch.y.detach().cpu())
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        return predictions, targets, all_outputs



def get_mean_min_ax(pred, targ):
    results = {}
    for idx,entry in enumerate(targ):
        try:
            results[entry.item()] = results[entry.item()] + [pred[idx].item()]
        except KeyError:
            results[entry.item()] = [pred[idx].item()]

    mean_data = np.array([[key, np.mean(results[key])] for key in results.keys()])
    max_data = np.array([[key, np.max(results[key])] for key in results.keys()])
    min_data = np.array([[key, np.min(results[key])] for key in results.keys()])

    sorted_indices = np.argsort(mean_data[:, 0])

    mean_data = mean_data[sorted_indices]
    max_data = max_data[sorted_indices]
    min_data = min_data[sorted_indices]
    return mean_data,max_data,min_data


folder = "UDN_RESULTS_300PATIENCE"
props = ["Diameter", "Eccentricity"] # , 'SSSP']
networks = ["adgn", "gcn"]

image_folder = "images"
if not os.path.exists(image_folder):
   os.makedirs(image_folder)


for prop in props:
    for network in networks:
        plt.cla()
        data_loader = get_test_data_loader(prop)
        model_udn = get_model(True, prop, network, folder)
        model = get_model(False, prop, network, folder)

        pred_udn, targ_udn, out_udn = run_model(model_udn, data_loader)
        mean_data_udn, max_data_udn, min_data_udn = get_mean_min_ax(pred_udn, targ_udn)

        pred, targ, out = run_model(model, data_loader)
        mean_data, max_data, min_data = get_mean_min_ax(pred, targ)


        plt.plot(np.arange(np.min(mean_data[:,0]),np.max(mean_data[:,0])),np.arange(np.min(mean_data[:,0]),np.max(mean_data[:,0])), label="Ideal", color='black', linewidth=2, dashes=[6, 2])

        plt.fill_between(mean_data_udn[:,0],min_data_udn[:,1],max_data_udn[:,1],alpha=0.2)
        plt.plot(mean_data_udn[:,0],mean_data_udn[:,1], label=r"\textsc{AMP}$_{\textsc{%s}}$" % network.upper())

        plt.fill_between(mean_data[:,0],min_data[:,1],max_data[:,1],alpha=0.2)
        plt.plot(mean_data[:,0],mean_data[:,1], label=r"\textsc{%s}" % network.upper())

        plt.xlabel(f"Target {prop.lower()}")
        plt.ylabel(f"Predicted {prop.lower()}")

        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{image_folder}/{prop}_{network}_targ_vs_pred.pdf")

        plt.cla()

        pred_udn_mean, targ_udn_mean, out_udn_mean = run_model(model_udn, data_loader, use_mean_layer=True)
        mean_data_udn_mean, max_data_udn_mean, min_data_udn_mean = get_mean_min_ax(pred_udn_mean, targ_udn_mean)

        plt.plot(np.arange(np.min(mean_data[:,0]),np.max(mean_data[:,0])),np.arange(np.min(mean_data[:,0]),np.max(mean_data[:,0])), color='black', linewidth=2, label="Ideal")

        plt.fill_between(mean_data_udn_mean[:,0],min_data_udn_mean[:,1],max_data_udn_mean[:,1],alpha=0.2)
        plt.plot(mean_data_udn_mean[:,0],mean_data_udn_mean[:,1], label=rf"Mean Layer")

        plt.fill_between(mean_data_udn[:,0],min_data_udn[:,1],max_data_udn[:,1],alpha=0.2)
        plt.plot(mean_data_udn[:,0],mean_data_udn[:,1], label=rf"Expectation Value")

        plt.xlabel("Target")
        plt.ylabel("Prediction")
        plt.title(prop)

        plt.legend()
        plt.savefig(f"{image_folder}/{prop}_{network}_mean_vs_expectation.pdf")

        plt.cla()

        sorting = targ_udn.argsort()
        targ_udn = targ_udn[sorting]
        indeces_for_target = {}
        for target_value in np.unique(targ_udn):
            indeces_for_target[target_value] = targ_udn==target_value


        qL = out_udn[0][2][5][0]

        outputs_0 = []
        for o in out_udn:
            outputs_0.append(o[0])
        outputs_0 = torch.cat(outputs_0,dim=0)
        print(outputs_0.shape)
        outputs_0 = outputs_0[sorting] #sort according to target
        max=torch.tensor(0)
        for target_value in np.unique(targ_udn)[::2]:
            averaged_for_target_value=outputs_0[indeces_for_target[target_value]].mean(axis=0)
            max = torch.maximum(torch.max(averaged_for_target_value), max)
            plt.plot(averaged_for_target_value, label=f"t={int(target_value)}")


        plt.plot(qL/torch.max(qL)*max, label="qL")
        plt.xlabel("Layer")
        plt.ylabel("qL or layerwise network output")

        plt.legend()
        plt.savefig(f"{image_folder}/{prop}_{network}_qL_vs_output.pdf")
