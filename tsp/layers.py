import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from . import typedef


class EdgeSum(MessagePassing):
    
    def __init__(self):
        super().__init__(flow="target_to_source", aggr="sum")
        
    def forward(self, e, edge_index):
        p = self.propagate(edge_index, e=e)
        return p[edge_index[0]]
    
    def message(self, e):
        return e


class NodeConv(MessagePassing):
    
    def __init__(self, units, eps=typedef.EPS):
        super().__init__(flow="target_to_source", aggr="sum")
        self.w_1 = nn.Linear(units, units, bias=False)
        self.w_2 = nn.Linear(units, units, bias=False)
        self.edge_sum = EdgeSum()
        self.eps = eps
    
    def forward(self, x, e, edge_index):
        e = e.sigmoid()
        e_sum = self.edge_sum(e, edge_index) + self.eps
        eta = e / e_sum
        return self.propagate(edge_index, x=x, eta=eta)
    
    def message(self, x_i, x_j, eta):        
        return self.w_1(x_i) + eta * self.w_2(x_j)
    

class NodeConvRes(nn.Module):
    
    def __init__(self, units, eps=typedef.EPS):
        super().__init__()
        self.node_conv = NodeConv(units, eps)
        self.bn = nn.BatchNorm1d(units)
        
    def forward(self, x, e, edge_index):
        h = self.node_conv(x, e, edge_index)
        h = self.bn(h).relu()
        return x + h
    

class EdgeLinrRes(MessagePassing):
    
    def __init__(self, units):
        super().__init__(flow="target_to_source")
        self.w_3 = nn.Linear(units, units, bias=False)
        self.w_4 = nn.Linear(units, units, bias=False)
        self.w_5 = nn.Linear(units, units, bias=False)
        self.bn = nn.BatchNorm1d(units)
        
    def forward(self, x, e, edge_index):
        return self.propagate(edge_index, x=x, e=e)
    
    def message(self, x_i, x_j, e):        
        h = self.w_3(e) + self.w_4(x_i) + self.w_5(x_j)
        h = self.bn(h).relu()
        return e + h
    
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        return inputs
