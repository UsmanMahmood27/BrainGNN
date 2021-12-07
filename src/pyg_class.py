


import torch
import torch.nn as nn
from torch_geometric.data import InMemoryDataset
import numpy as np
import pandas as pd
import pickle
import csv
import os
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv, ASAPooling, SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, GlobalAttention as GA, Set2Set as Set
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing, ChebConv, GCNConv, GATConv
from torch_geometric.transforms import LaplacianLambdaMax
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.data import Batch as Batch
import  time
import math
from torch.nn import Parameter


class pyg_data_creation(InMemoryDataset):
    def __init__(self, root, dataset, file_name="dataset", transform=None, pre_transform=None):
        self.file_name = file_name
        self.dataset = dataset
        super(pyg_data_creation, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['../input/'+self.file_name+'.dataset']

    def download(self):
        pass
    
    def process(self):
        
        data_list = []
        # self.data_tensor = self.data_tensor.reshape(self.data_tensor.shape[0] * self.data_tensor.shape[1], -1).contiguous()

        # process by session_id
        # grouped = df.groupby('session_id')
        for subject, label in self.dataset:
            n_nodes = subject.shape[0]
            node_features = subject.clone()
            edge_index = torch.from_numpy(np.arange(0, n_nodes)).int()
            a = (edge_index.unsqueeze(0).repeat(n_nodes, 1))
            b =  (edge_index.unsqueeze(1).repeat(1, n_nodes))
            edge_index = torch.cat((edge_index.unsqueeze(1).repeat(1, n_nodes).reshape(1,-1),
                                   edge_index.unsqueeze(0).repeat(n_nodes, 1).reshape(1,-1)),dim=0).long()

            x = node_features

            y = label.float().view(1)

            data = Data(x=x, edge_index=edge_index, y=y, edge_attr="")
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        








class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr='max') #  "Max" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(in_channels + out_channels, in_channels, bias=False)
        self.update_act = torch.nn.ReLU()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]


        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))


        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        # x_j has shape [E, in_channels]

        x_j = self.lin(x_j)
        x_j = self.act(x_j)

        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]


        new_embedding = torch.cat([aggr_out, x], dim=1)

        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)

        return new_embedding


embed_dim = 24


class Net(torch.nn.Module):
    def __init__(self, n_regions=116):
        super(Net, self).__init__()

        self.attn = nn.Sequential(
            nn.Linear(130, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.conv1 = GatedGraphConv(embed_dim, 2, aggr='add', bias=True)

        self.final_size = 32
        self.size1=n_regions
        self.size2=math.ceil(self.size1 * 0.8)
        self.size3 = math.ceil(self.size2 * 0.8)
        self.size4 = math.ceil(self.size3 * 0.3)
        self.pool1 = TopKPooling(embed_dim, ratio=0.8)
        self.gp1 = Set(embed_dim, 2, 1)
        self.conv2 = GatedGraphConv(embed_dim, 2, aggr='add', bias=True)
        self.pool2 = TopKPooling(embed_dim, ratio=0.8)
        self.gp2 = Set(embed_dim, 2, 1)
        self.conv3 = GatedGraphConv(embed_dim, 2, aggr='add', bias=True)
        self.pool3 = TopKPooling(embed_dim, ratio=0.3)
        self.gp3 = Set(embed_dim, 2, 1)

        self.conv4 = GatedGraphConv(embed_dim, 2, aggr='add', bias=True)
        self.gp4 = Set(embed_dim, 2, 1)
        #
        self.conv5 = GatedGraphConv(embed_dim, 2, aggr='add', bias=True)
        self.gp5 = Set(embed_dim, 2, 1)
        #
        self.conv6 = GatedGraphConv(embed_dim, 2, aggr='add', bias=True)
        self.gp6 = Set(embed_dim, 2, 1)


        self.lin1 = torch.nn.Linear(96, 32)
        self.bn1 = torch.nn.BatchNorm1d(96)
        self.act1 = torch.nn.ReLU()






    def forward(self, data, epoch=0):

        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        B=data.y.shape[0]

        indices = torch.from_numpy(np.arange(116))
        indices = indices.unsqueeze(dim=0).repeat(B,1).reshape(B*116)


        x = F.relu(self.conv1(x, edge_index,torch.squeeze(edge_attr))  )
        x, edge_index, edge_attr, batch, perm, score_perm = self.pool1(x, edge_index, edge_attr, batch)

        x = F.relu(self.conv2(x, edge_index, torch.squeeze(edge_attr))  )
        x, edge_index, edge_attr, batch, perm, score_perm = self.pool2(x, edge_index, edge_attr, batch)

        x = F.relu(self.conv3(x, edge_index, torch.squeeze(edge_attr))  )
        x, edge_index, edge_attr, batch, perm, score_perm = self.pool3(x, edge_index, edge_attr, batch)

        x = F.relu(self.conv4(x, edge_index, torch.squeeze(edge_attr)) )
        x = F.relu(self.conv5(x, edge_index, torch.squeeze(edge_attr)) )
        x = F.relu(self.conv6(x, edge_index, torch.squeeze(edge_attr)) )
        x6 = torch.cat([gmp(x, batch), gap(x, batch), self.gp6(x, batch)], dim=1)

        x = x6
        x = self.lin1(x)
        x = self.act1(x)


        return x, indices










