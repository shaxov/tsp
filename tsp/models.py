import torch
import torch.nn as nn
from .layers import NodeConvRes, EdgeLinrRes

from . import typedef


class ConvNet(nn.Module):
    
    def __init__(self, units: int, knn_dim: int = None, num_layers=1, eps=typedef.EPS):
        super().__init__()
        self.knn = knn_dim
        if units % 2 != 0:
            raise ValueError("'embedding_dim' must be even.")
        assert num_layers >= 1

        half_units = units // 2
        self.node_linear = nn.Linear(2, units)
        self.dist_linear = nn.Linear(1, half_units if knn_dim else units)
        if knn_dim:
            self.knns_linear = nn.Linear(knn_dim, half_units, bias=False)
        self.node_convs = nn.ModuleList([
            NodeConvRes(units, eps) for _ in range(num_layers)])
        self.edge_linrs = nn.ModuleList([
            EdgeLinrRes(units) for _ in range(num_layers + 1)])
        
        self.mlp = nn.Sequential(
            nn.Linear(units, half_units),
            nn.PReLU(),
            nn.Linear(half_units, 1),
        )
        
    def forward(self, data):
        node_pos = data.pos
        edge_knn = data.knn_label
        edge_dist = data.edge_attr.view(-1, 1)
        edge_index = data.edge_index

        x = self.node_linear(node_pos)
        e = self.dist_linear(edge_dist)
        if self.knn:
            e = torch.cat([e, self.knns_linear(edge_knn)], dim=-1)

        for node_conv, edge_linr in zip(self.node_convs, self.edge_linrs[:-1]):
            x = node_conv(x, e, edge_index)
            e = edge_linr(x, e, edge_index)
        e = self.edge_linrs[-1](x, e, edge_index)
        return self.mlp(e)
