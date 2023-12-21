import torch
import math
import torch_sparse
import deeprobust.graph.utils as utils
from torch import nn
from torch_sparse import SparseTensor

# SGC
class SGC(nn.Module):
    
    # 双隐藏层
    def __init__(self, nfeat, nhid, nclass, hop=2, nlayers=2, dropout=0.5, with_relu=True, 
                 with_bias=True, with_bn=None, pre=False, post=False):

        super().__init__()
        self.weight1 = nn.Parameter(torch.FloatTensor(nfeat, nhid))
        self.bias1 = nn.Parameter(torch.FloatTensor(nhid)) if with_bias else None
        self.weight2 = nn.Parameter(torch.FloatTensor(nhid, nclass))
        self.bias2 = nn.Parameter(torch.FloatTensor(nclass)) if with_bias else None
        self.hop = hop
        self.dropout = dropout
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.pre = pre
        self.post = post
        self.initialize()

    def initialize(self):

        stdv1 = 1. / math.sqrt(self.weight1.T.size(1))
        stdv2 = 1. / math.sqrt(self.weight2.T.size(1))
        self.weight1.data.uniform_(-stdv1, stdv1)
        self.weight2.data.uniform_(-stdv2, stdv2)
        if self.with_bias:
            self.bias1.data.uniform_(-stdv1, stdv1)
            self.bias2.data.uniform_(-stdv2, stdv2)

    def forward(self, x, adj):

        if self.pre is True:
            for i in range(self.hop):
                if isinstance(adj, torch_sparse.SparseTensor):
                    x = torch_sparse.matmul(adj, x)
                else:
                    x = torch.mm(adj, x)
        x = torch.mm(x, self.weight1)
        x += self.bias1 if self.with_bias else 0
        if self.with_relu:
            x = torch.relu(x)
        x = nn.functional.dropout(x, self.dropout, training=self.training)
        
        x = torch.mm(x, self.weight2)
        x += self.bias2 if self.with_bias else 0
        
        if self.post is True:
            for i in range(self.hop):
                if isinstance(adj, torch_sparse.SparseTensor):
                    x = torch_sparse.matmul(adj, x)
                else:
                    x = torch.mm(adj, x)
    
        return nn.functional.log_softmax(x, dim=1) 

#GraphSage
class SageConvolution(nn.Module):

    def __init__(self, in_features, out_features, with_bias=True, root_weight=False):
        
        super(SageConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_l = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias_l = nn.Parameter(torch.FloatTensor(out_features))
        self.weight_r = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias_r = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        self.root_weight = root_weight

    def reset_parameters(self):
        
        stdv = 1. / math.sqrt(self.weight_l.T.size(1))
        self.weight_l.data.uniform_(-stdv, stdv)
        self.bias_l.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weight_r.T.size(1))
        self.weight_r.data.uniform_(-stdv, stdv)
        self.bias_r.data.uniform_(-stdv, stdv)
        
    def forward(self, input, adj, size=None):

        output = torch.mm(input, self.weight_l)
        if isinstance(adj, torch_sparse.SparseTensor):
            output = torch_sparse.matmul(adj, output)
        else:
            output = torch.spmm(adj, output)
        output = output + self.bias_l

        if self.root_weight:
            if size is not None:
                output = output + input[:size[1]] @ self.weight_r + self.bias_r
            else:
                output = output + input @ self.weight_r + self.bias_r
        else:
            output = output

        return output

class GraphSage(nn.Module):

    def __init__(self, nfeat, nhid, nclass, hop=None, nlayers=2, dropout=0.5, 
                 with_relu=True, with_bias=True, with_bn=False, pre=None, post=None):

        super(GraphSage, self).__init__()

        self.nfeat = nfeat
        self.nclass = nclass

        self.layers = nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(SageConvolution(nfeat, nclass, with_bias=with_bias))
        else:
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(SageConvolution(nfeat, nhid, with_bias=with_bias))
            for i in range(nlayers-2):
                self.layers.append(SageConvolution(nhid, nhid, with_bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(SageConvolution(nhid, nclass, with_bias=with_bias))
        self.dropout = dropout
        self.with_relu = with_relu
        self.with_bn = with_bn
        self.with_bias = with_bias

    def forward(self, x, adj):
        
        for ix, layer in enumerate(self.layers):
            x = layer(x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = torch.relu(x)
                x = nn.functional.dropout(x, self.dropout, training=self.training)
                
        return nn.functional.log_softmax(x, dim=1)
    
    def forward_sampler(self, x, adjs):

        for ix, (adj, _, size) in enumerate(adjs):
            x = self.layers[ix](x, adj, size=size)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = torch.relu(x)
                x = nn.functional.dropout(x, self.dropout, training=self.training)

        return nn.functional.log_softmax(x, dim=1)

    def initialize(self):

        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()