import torch
from tqdm import tqdm
import torch_cluster as pyc
from typing import Union, List
from torchvision.transforms import Compose


class _FilterTransform:

    def __init__(self, condition: callable):
        self.condition = condition

    def __call__(self, data):
        return list(filter(self.condition, data))


class MaxNodesTransform(_FilterTransform):

    def __init__(self, max_nodes: int):
        super().__init__(condition=lambda entry: len(entry.pos) <= max_nodes)


class KNNTransform:

    def __init__(self, k: Union[int, List[int]]):
        self.k = k
        if isinstance(k, int):
            self.k = [k]

    @staticmethod
    def _edge_index_split(edge_index):
        source, target = edge_index
        indices = tuple([
            int((source == i).sum())
            for i in torch.unique(source)
        ])
        return torch.split(target, indices)

    def __call__(self, data):
        split = KNNTransform._edge_index_split
        for entry in tqdm(data, desc="KNNTransform", unit="graph"):
            knn_label_list = []
            entry_split = split(entry.edge_index)
            if len(entry.edge_attr.shape) == 1:
                entry.edge_attr = entry.edge_attr.view(-1, 1)
            for k in self.k:
                knn_split = split(pyc.knn(entry.pos, entry.pos, k + 1))
                knn_label = torch.cat([
                    torch.isin(x_i, x_j[1:]) for x_i, x_j in zip(entry_split, knn_split)],
                    dim=0,
                ).unsqueeze(-1).float()
                knn_label_list.append(knn_label)
            knn_attr = torch.cat(knn_label_list, dim=-1)
            entry.edge_attr = torch.cat([entry.edge_attr, knn_attr], dim=-1)
        return data


def get(name):
    return {
        "max_nodes_transform": MaxNodesTransform,
        "knn_transform": KNNTransform,
    }[name]
