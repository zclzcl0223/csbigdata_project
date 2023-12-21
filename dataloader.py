import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.preprocessing import StandardScaler
from torch_geometric import transforms as T

class DataLoader():
    """
    统一化数据加载
    """
    def __init__(self, name):
        path = './' + 'data/' + name
        if name in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(root=path, name=name, transform=T.NormalizeFeatures())
        elif name in ['ogbn-products']:
            dataset = PygNodePropPredDataset(root=path, name=name)
        else:
            raise NotImplementedError

        data = dataset[0]
        self.dataset = name
        self.num_nodes = data.num_nodes
        self.num_features = data.x.shape[1]
        self.num_classes = data.y.max().item() + 1
        
        if name in ['ogbn-products']:
            split_idx = dataset.get_idx_split()
            self.train_idx = split_idx['train']
            self.val_idx = split_idx['valid']
            self.test_idx = split_idx['test']
            self.edge_index = data.edge_index
        else:
            self.train_idx = torch.nonzero(data.train_mask).flatten()
            self.val_idx = torch.nonzero(data.val_mask).flatten()
            self.test_idx = torch.nonzero(data.test_mask).flatten()
            self.edge_index = data.edge_index

        self.num_edges = self.edge_index.shape[1]
        # 大图用训练集归一化
        if name in ['ogbn-products']:
            train_nodes = data.x[self.train_idx]
            scaler = StandardScaler()
            scaler.fit(train_nodes)
            self.x = scaler.transform(data.x)
        else:
            self.x = data.x
        self.y = data.y.reshape(-1)
        
        self.full_adj = sp.csr_matrix((np.ones(self.edge_index.shape[1]),
                                      (self.edge_index[0], self.edge_index[1])), 
                                      shape=(self.num_nodes, self.num_nodes))  # 稀疏化
        self.train_adj = self.full_adj[np.ix_(self.train_idx, self.train_idx)]  # 训练集子图
        self.val_adj = self.full_adj[np.ix_(self.val_idx, self.val_idx)]  # 训练集子图
        self.test_adj = self.full_adj[np.ix_(self.test_idx, self.test_idx)]  # 测试集子图
        
    def info(self):
        
        print('%s Train nodes: %d, Val nodes: %d, Test nodes: %d\n' % 
              (self.dataset, len(self.train_idx), len(self.val_idx), len(self.test_idx)))
        print('\tTrain adj: %d, Val adj: %d, Test adj: %d\n' %
              (self.train_adj.nnz/2, self.val_adj.nnz/2, self.test_adj.nnz/2))