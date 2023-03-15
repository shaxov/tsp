import torch
from . import typedef
from typing import Union, Tuple
from torch.nn import Module, Linear, BatchNorm1d
from torch_geometric.nn import MessagePassing


class EdgeAttention(MessagePassing):
    
    def __init__(self, eps=typedef.EPS):
        super().__init__(aggr="add")
        self.eps = eps
        
    def forward(self, edge_attr, edge_index):
        edge_attr = torch.sigmoid(edge_attr)
        prop_attr = self.propagate(edge_index, edge_attr=edge_attr)
        prop_attr = prop_attr[edge_index[1]] + self.eps
        return edge_attr / prop_attr
       
    def message(self, edge_attr):
        return edge_attr
    

class GEAConv(MessagePassing):

    """ Graph Edge Attention Convolution. """

    def __init__(self, in_channels, out_channels, eps=typedef.EPS):
       super().__init__(aggr="add")
       self.lin = Linear(in_channels, out_channels, bias=False)
       self.edge_atten = EdgeAttention(eps=eps)

    def forward(self, x, edge_attr, edge_index):
        edge_atten = self.edge_atten(edge_attr, edge_index)
        return self.propagate(edge_index, x=x, edge_atten=edge_atten)

    def message(self, x_j, edge_atten):        
        return edge_atten * self.lin(x_j)
    

class GEA2Conv(GEAConv):

    def __init__(self, in_channels, out_channels, eps=typedef.EPS):
       super().__init__(in_channels, out_channels, eps)
       self.lin = Linear(in_channels * 2, out_channels, bias=False)

    def message(self, x_i, x_j, edge_atten):        
        return edge_atten * self.lin(torch.cat([x_i, x_j - x_i], dim=-1))
    

class ResNodeConv(Module):

    def __init__(self,
                 in_channels: int, 
                 out_channels: int):
        super().__init__()
        self.gea_conv = GEAConv(in_channels, out_channels)
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bn = BatchNorm1d(out_channels)
    
    def forward(self, x, edge_attr, edge_index):
        h = self.lin(x) + self.gea_conv(x, edge_attr, edge_index)
        h = torch.relu(self.bn(h))
        return x + h
    

class ResNode2Conv(ResNodeConv):

    def __init__(self,
                 in_channels: int, 
                 out_channels: int):
        super().__init__(in_channels, out_channels)
        self.gea_conv = GEA2Conv(in_channels, out_channels)
    

class ResEdgeLinear(MessagePassing):
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int):
        super().__init__()
        self.lin1 = Linear(in_features, out_features, bias=False)
        self.lin2 = Linear(in_features, out_features, bias=False)
        self.lin3 = Linear(in_features, out_features, bias=False)
        self.bn = BatchNorm1d(out_features)
        
    def forward(self, x, edge_attr, edge_index):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):        
        h = self.lin1(edge_attr) + self.lin2(x_i) + self.lin3(x_j)
        h = torch.relu(self.bn(h))
        return edge_attr + h
    
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        return inputs
