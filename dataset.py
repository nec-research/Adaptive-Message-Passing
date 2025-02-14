import itertools
import os
import random
from typing import Union, List, Tuple, Optional, Callable

import math
import numpy as np
import torch
import torch_geometric
import torchvision
from pydgn.data.dataset import DatasetInterface
from torch.utils.data import ConcatDataset
from torch_geometric.data import Data
from torchvision import transforms


class CIFAR10(DatasetInterface):
    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(root, name, transform, pre_transform, pre_filter)
        self.train_data = torch.load(self.processed_paths[0])
        self.test_data = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ["train_data.pt", "test_data.pt"]

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, "processed")

    def download(self):
        pass

    def process(self):
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        train_set = torchvision.datasets.CIFAR10(
            root=self.raw_dir,
            train=True,
            download=True,
            transform=transform_train,
        )

        test_set = torchvision.datasets.CIFAR10(
            root=self.raw_dir,
            train=False,
            download=True,
            transform=transform_test,
        )

        print(f"Train set length: {len(train_set)}")
        print(f"Test set length: {len(test_set)}")

        torch.save(train_set, self.processed_paths[0])
        torch.save(test_set, self.processed_paths[1])

    def get(self, idx: int) -> Data:
        len_train, len_test = len(self.train_data), len(self.test_data)

        if idx < len_train:
            sample = self.train_data[idx]
        else:
            sample = self.test_data[idx - len_train]

        s = Data(
            x=sample[0].unsqueeze(0),  # add node dimension
            y=torch.tensor([sample[1]], dtype=torch.long),
        )
        return s

    @property
    def dim_node_features(self) -> int:
        return 784

    @property
    def dim_edge_features(self) -> int:
        return 0

    @property
    def dim_target(self) -> int:
        return 10

    def __len__(self) -> int:
        return len(self.train_data) + len(self.test_data)


class GraphPropertyPrediction(DatasetInterface):

    def __init__(
        self,
        root: str,
        name: str,
        dim='25-35',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        **kwargs,
    ):
        TASKS = ['dist', 'ecc', 'diam']

        self.name_file_dict = {
            'SSSP': 'dist',
            'Diameter': 'diam',
            'Eccentricity': 'ecc',
        }

        assert transform == None
        assert pre_transform == None
        assert pre_filter == None

        super().__init__(root, name, transform, pre_transform, pre_filter)

        assert self.name_file_dict[self.name] in TASKS
        assert dim in ['25-35']

        # target has already been normalized by authors of ADGN paper
        self.data_list = torch.load(self.raw_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        r"""
        The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading.
        """
        return [f'{self.name_file_dict[self.name]}_25-35_data_list.pt']

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self.raw_file_names

    def download(self):
        raise NotImplementedError('you should already provide the raw files '
                                  f'in the folder {self.root}/{self.name}/raw')

    def process(self):
        pass

    def get(self, idx: int) -> Data:
        r"""
        Gets the data object at index :obj:`idx`.
        """
        return self.data_list[idx]

    @property
    def dim_node_features(self) -> int:
        r"""
        Specifies the number of node features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        return self.data_list[0].x.shape[1]

    @property
    def dim_edge_features(self) -> int:
        r"""
        Specifies the number of edge features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        return 0

    @property
    def dim_target(self) -> int:
        r"""
        Specifies the dimension of each target vector.
        """
        return 1

    def len(self) -> int:
        r"""
        Returns the number of graphs stored in the dataset.
        Note: we need to implement both `len` and `__len__` to comply with
        PyG interface
        """
        return len(self)

    def __len__(self) -> int:
        return len(self.data_list)


class QM9(torch_geometric.datasets.QM9):
    TASKS = ["mu", "alpha", "HOMO", "LUMO", "gap", "R2", "ZPVE", "U0", "U",
             "H", "G", "Cv", "Omega"]

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        **kwargs,
    ):
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def dim_node_features(self) -> int:
        r"""
        Specifies the number of node features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        return self.get(0).x.shape[1]

    @property
    def dim_edge_features(self) -> int:
        r"""
        Specifies the number of edge features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        return 4

    @property
    def dim_target(self) -> int:
        r"""
        Specifies the dimension of each target vector.
        """
        return self.get(0).y.shape[1]


class QM9mu(QM9):
    def get(self, idx: int) -> Data:
        d = super().get(idx)
        return Data(x=d.x,
                    edge_attr=d.edge_attr.argmax(1),
                    edge_index=d.edge_index,
                    y=d.y[:, 0])
    @property
    def dim_target(self) -> int:
        return 1


class QM9alpha(QM9):
    def get(self, idx: int) -> Data:
        d = super().get(idx)
        return Data(x=d.x,
                    edge_attr=d.edge_attr.argmax(1),
                    edge_index=d.edge_index,
                    y=d.y[:, 1])
    @property
    def dim_target(self) -> int:
        return 1


class Peptides(DatasetInterface):

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        **kwargs,
    ):
        self.name = name
        super().__init__(root, name, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])

    def download(self):
        pass

    def process(self):
        pass

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []  # you should already have processed this

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    @property
    def dim_node_features(self) -> int:
        r"""
        Specifies the number of node features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        return self.get(0).x.shape[1]

    @property
    def dim_edge_features(self) -> int:
        r"""
        Specifies the number of edge features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        return self.get(0).edge_attr.shape[1]

    @property
    def dim_target(self) -> int:
        r"""
        Specifies the dimension of each target vector.
        """
        return self.get(0).y.shape[1]

    def get(self, idx: int) -> Data:
        d = self.data[idx]
        d.x = d.x.float()
        return d

    def len(self) -> int:
        r"""
        Returns the number of graphs stored in the dataset.
        Note: we need to implement both `len` and `__len__` to comply with
        PyG interface
        """
        return len(self)

    def __len__(self):
        return len(self.data)


class TreeNeighborsMatch(DatasetInterface):
    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        **kwargs):

        self.name = name
        assert name in ['TNM_2', 'TNM_3', 'TNM_4', 'TNM_5', 'TNM_6']
        self.depth = int(name.split(sep='_')[1])
        self.num_nodes, self.edges, self.leaf_indices = self._create_blank_tree()

        super().__init__(root, name, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])
        self.n_c = max([self.data[i].y for i in range(len(self))])

    def download(self):
        pass

    def process(self):
        data_list = self.generate_data()
        torch.save(data_list, self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []  # you should already have processed this

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    @property
    def dim_node_features(self) -> int:
        r"""
        Specifies the number of node features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        return self.get(0).x.shape[1]

    @property
    def dim_edge_features(self) -> int:
        r"""
        Specifies the number of edge features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        return 0

    @property
    def dim_target(self) -> int:
        r"""
        Specifies the dimension of each target vector.
        """
        return self.n_c

    def get(self, idx: int) -> Data:
        d = self.data[idx]
        d.x = d.x.float()
        d.y = d.y - 1
        return d

    def len(self) -> int:
        r"""
        Returns the number of graphs stored in the dataset.
        Note: we need to implement both `len` and `__len__` to comply with
        PyG interface
        """
        return len(self)

    def __len__(self):
        return len(self.data)

    def add_child_edges(self, cur_node, max_node):
        edges = []
        leaf_indices = []
        stack = [(cur_node, max_node)]
        while len(stack) > 0:
            cur_node, max_node = stack.pop()
            if cur_node == max_node:
                leaf_indices.append(cur_node)
                continue
            left_child = cur_node + 1
            right_child = cur_node + 1 + ((max_node - cur_node) // 2)
            edges.append([left_child, cur_node])
            edges.append([right_child, cur_node])
            stack.append((right_child, max_node))
            stack.append((left_child, right_child - 1))
        return edges, leaf_indices

    def _create_blank_tree(self):
        max_node_id = 2 ** (self.depth + 1) - 2
        edges, leaf_indices = self.add_child_edges(cur_node=0, max_node=max_node_id)
        return max_node_id + 1, edges, leaf_indices

    def create_blank_tree(self, add_self_loops=True):
        edge_index = torch.tensor(self.edges).t()
        if add_self_loops:
            edge_index, _ = torch_geometric.utils.add_remaining_self_loops(edge_index=edge_index, )
        return edge_index

    def generate_data(self):
        data_list = []

        for comb in self.get_combinations():
            edge_index = self.create_blank_tree(add_self_loops=True)
            nodes = torch.tensor(self.get_nodes_features(comb), dtype=torch.long)
            root_mask = torch.tensor([True] + [False] * (len(nodes) - 1))
            label = torch.tensor([self.label(comb)], dtype=torch.long)
            data_list.append(Data(x=nodes, edge_index=edge_index, root_mask=root_mask, y=label))

        print(f'Generated {len(data_list)} graphs...')
        return data_list

    def get_combinations(self):
        # returns: an iterable of [key, permutation(leaves)]
        # number of combinations: (num_leaves!)*num_choices
        num_leaves = len(self.leaf_indices)
        num_permutations = 1000
        max_examples = 32000

        if self.depth > 3:
            per_depth_num_permutations = min(num_permutations, math.factorial(num_leaves), max_examples // num_leaves)
            permutations = [np.random.permutation(range(1, num_leaves + 1)) for _ in
                            range(per_depth_num_permutations)]
        else:
            permutations = random.sample(list(itertools.permutations(range(1, num_leaves + 1))),
                                         min(num_permutations, math.factorial(num_leaves)))

        return itertools.chain.from_iterable(

            zip(range(1, num_leaves + 1), itertools.repeat(perm))
            for perm in permutations)

    def get_nodes_features(self, combination):
        # combination: a list of indices
        # Each leaf contains a one-hot encoding of a key, and a one-hot encoding of the value
        # Every other node is empty, for now
        selected_key, values = combination

        # The root is [one-hot selected key] + [0 ... 0]
        nodes = [ (selected_key, 0) ]

        for i in range(1, self.num_nodes):
            if i in self.leaf_indices:
                leaf_num = self.leaf_indices.index(i)
                node = (leaf_num+1, values[leaf_num])
            else:
                node = (0, 0)
            nodes.append(node)
        return nodes

    def label(self, combination):
        selected_key, values = combination
        return int(values[selected_key - 1])

    def get_dims(self):
        # get input and output dims
        in_dim = len(self.leaf_indices)
        out_dim = len(self.leaf_indices)
        return in_dim, out_dim