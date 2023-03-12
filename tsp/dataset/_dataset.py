import torch
import torch_cluster as pyc
import torch_geometric as pyg


class TSPDataset(torch.utils.data.Dataset):
    
    def __init__(self, root: str = "./tsp_dataset", split: str = "train", transforms=None):
        self.root = root
        self.split = split
        self._data = pyg.datasets.GNNBenchmarkDataset(
            root=root, name="TSP", split=split)
        if transforms:
            self._data = transforms(self._data)

    def apply_transform(self, transform):
        self._data = transform(self._data)
        return self
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        return self._data[idx]
    
    def __repr__(self):
        return f"TSPDataset(root={self.root}, split={self.split}, size={len(self)})"
