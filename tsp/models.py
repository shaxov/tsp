import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import EdgeConv
from .layers import ResNodeConv, ResNode2Conv, ResEdgeLinear


class ConvNet(nn.Module):

    """ Convolutional network for Traveling Salesman Problem."""
    
    def __init__(
            self, 
            units: int, 
            num_layers: int = 1, 
            knn_dim: int = 0,
        ):
        super().__init__()
        if units % 2 != 0:
            raise ValueError("'units' must be even.")
        assert num_layers >= 1

        half_units = units // 2
        self.lin_pos = nn.Linear(2, units)
        self.lin_attr = nn.Linear(1 + knn_dim, units)

        self.res_node_conv_list = nn.ModuleList([
            ResNodeConv(units, units) for _ in range(num_layers)])
        self.res_edge_linear_list = nn.ModuleList([
            ResEdgeLinear(units, units) for _ in range(num_layers + 1)])
        
        self.mlp = nn.Sequential(
            nn.Linear(units + 1, half_units),
            nn.PReLU(),
            nn.Linear(half_units, 1),
        )
        
    def forward(self, data):
        x = self.lin_pos(data.pos)
        edge_attr = self.lin_attr(data.edge_attr)

        handle = zip(self.res_node_conv_list, self.res_edge_linear_list[:-1])
        for res_node_conv, res_edge_linear in handle:
            x, edge_attr = (
                res_node_conv(x, edge_attr, data.edge_index), 
                res_edge_linear(x, edge_attr, data.edge_index),
            )
        edge_attr = self.res_edge_linear_list[-1](x, edge_attr, data.edge_index)
        logits = self.mlp(torch.cat([edge_attr, data.edge_attr[:, [0]]], dim=-1))
        return logits
    

class Conv2Net(ConvNet):
    
    def __init__(
            self, 
            units: int, 
            num_layers: int = 1, 
            knn_dim: int = 0,
        ):
        super().__init__(units, num_layers, knn_dim)
        self.res_node_conv_list = nn.ModuleList([
            ResNode2Conv(units, units) for _ in range(num_layers)]) 


class _MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, dropout=0.):
        super().__init__()
        self.rate = dropout
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.p_relu = nn.PReLU()

    def forward(self, x):
        x = self.lin1(x)
        x = self.p_relu(x)
        x = F.dropout(x, p=self.rate, training=self.training)
        x = self.lin2(x)
        return x


class TSPNet(torch.nn.Module):

    def __init__(self, embedding_dim=64, num_layers=3, dropout=0.):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.edge_conv_list = []
        for i in range(num_layers):
            in_channels = embedding_dim
            if i == 0:
                in_channels = 2
            self.edge_conv_list.append(
                EdgeConv(
                    _MLP(in_channels=2*in_channels,
                        out_channels=embedding_dim,
                        hidden_channels=embedding_dim,
                        dropout=dropout), aggr="max"))
        self.edge_conv_list = nn.ModuleList(self.edge_conv_list)
        # self.classifier = nn.Linear(2*embedding_dim + 1, 1)
        self.classifier = _MLP(in_channels=2*embedding_dim + 1, out_channels=1, hidden_channels=256)

    def forward(self, data):
        x = data.pos
        for edge_conv in self.edge_conv_list:
            x = edge_conv(x, data.edge_index)
            x = x.relu()

        x_src = x[data.edge_index[0,:],:]
        x_trg = x[data.edge_index[1,:],:]
        x = torch.cat([x_src, x_trg, data.edge_attr.view(-1, 1)], dim=1)
        return self.classifier(x)[:, 0]


def get(name):
    return {
        "conv_net": ConvNet,
        "conv2_net": Conv2Net,
        "tsp_net": TSPNet,
    }[name]
