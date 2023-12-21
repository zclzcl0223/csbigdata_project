import torch
import numpy as np
import scipy.sparse as sp
import deeprobust.graph.utils as utils
import matplotlib.pyplot as plt
from torch_sparse import SparseTensor
from matplotlib_inline import backend_inline
from torch_geometric.loader import NeighborSampler
from IPython import display

def data_convert(data, adj, device, idx=None, normalize=True):
    
    if idx is None:
        feat = data.x
        label = data.y
    else:
        feat = data.x[idx]
        label = data.y[idx]
    feat, adj, label = utils.to_tensor(feat, adj, label)
    feat = feat.to(device)
    adj = adj.to(device)

    label = label.to(device)
    # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    if normalize:
        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_adj_tensor(adj)
        adj = adj_norm
    adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
            value=adj._values(), sparse_sizes=adj.size()).t()
    
    return feat, adj, label

def get_train_loader(adj, node_idx):

    if adj.density() > 0.5: # if the weighted graph is too dense, we need a larger neighborhood size
        sizes = [30, 20]
    else:
        sizes = [20, 15]
    train_loader = NeighborSampler(adj, node_idx=node_idx,
                                    sizes=sizes, batch_size=len(node_idx),
                                    num_workers=0, return_e_id=False,
                                    num_nodes=adj.size(0),
                                    shuffle=True)

    return train_loader

def get_syn_adj(feat, alpha, A, E):
    
    XXT = torch.mm(feat, feat.T)
    syn_adj = torch.mm(torch.inverse(XXT + alpha * E), (XXT + alpha * A))
    
    return syn_adj

class Animator:
    
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 3)):
        
        if legend is None:
            legend = []
        backend_inline.set_matplotlib_formats('svg')
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
    
    def add(self, x, y):
        
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i] .append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim), axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()