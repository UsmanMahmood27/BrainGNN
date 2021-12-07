import torch
import torch.nn as nn
import torch.nn.utils.rnn as tn
from torch.autograd import Variable
import os
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch.utils.data import TensorDataset
import scipy
import time
import copy
from .utils import init
from torch.nn import Parameter

class combinedModel(nn.Module):
    """Bidirectional LSTM for classifying subjects."""

    def __init__(self, encoder, graph, samples_per_subject, gain=0.1, PT="", exp="UFPT", device="cuda", oldpath=""):

        super().__init__()
        self.encoder = encoder
        # self.lstm = lstm
        self.gain = 0.25
        self.samples_per_subject = samples_per_subject
        self.graph = graph
        self.PT = PT
        self.exp = exp
        self.device = device
        self.oldpath=oldpath
        self.n_heads=1

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        # self.attn = nn.Sequential(
        #     nn.Linear(2 * self.lstm.hidden_dim , 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1)
        # ).to(device)

        # self.classifier_weights = nn.Sequential(
        #     #nn.BatchNorm1d(300),
        #
        #     #nn.BatchNorm1d(128),
        #     nn.Linear(48, 16),
        #     nn.ReLU(),
        #     #nn.BatchNorm1d(16),
        #     nn.Linear(16, 1),
        #     #nn.ReLU()
        # ).to(device)

        self.decoder = nn.Sequential(
            nn.Linear(self.graph.final_size, 2),

        ).to(device)

        self.encoder_decoder = nn.Sequential(
            nn.Linear(7424, 1024), #115200
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 2)

        ).to(device)

        # self.lstm_decoder = nn.Sequential(
        #     nn.Linear(13920, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 2)
        #
        # ).to(device)

        # self.decoder2 = nn.Sequential(
        #     nn.Linear(3328, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 2),
        #
        # ).to(device)



        self.key_layer = nn.Sequential(
            nn.Linear(samples_per_subject * self.encoder.feature_size , samples_per_subject * 24),
        ).to(device)

        self.value_layer = nn.Sequential(
            nn.Linear(samples_per_subject * self.encoder.feature_size , samples_per_subject * 24),
        ).to(device)

        self.query_layer = nn.Sequential(
            nn.Linear(samples_per_subject * self.encoder.feature_size , samples_per_subject * 24),
        ).to(device)

        self.multihead_attn = nn.MultiheadAttention(samples_per_subject * 24, self.n_heads).to(self.device)

        self.classifier1 = nn.Sequential(
            nn.Linear(24, self.graph.final_size),
            nn.ReLU(),
        ).to(device)


    def init_weight(self, PT="UFPT"):
        print(self.gain)
        print('init' + PT)
        if PT == "NPT":
            for name, param in self.query_layer.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param, gain=self.gain)
            for name, param in self.key_layer.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param, gain=self.gain)
            for name, param in self.value_layer.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param, gain=self.gain)
            for name, param in self.multihead_attn.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param, gain=self.gain)
            for name, param in self.graph.named_parameters():
                if 'weight' in name:
                    if 'bn1.weight' not in name and 'bn2.weight' not in name and 'bn3.weight' not in name:
                        # print(name)
                        nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.decoder.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        # for name, param in self.classifier_weights.named_parameters():
        #    if 'weight' in name:
        #         nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.encoder_decoder.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)


    def loadModels(self):
        if self.PT in ['milc-fMRI', 'variable-attention', 'two-loss-milc']:
            if self.exp in ['UFPT', 'FPT']:
                print('in ufpt and fpt')
                model_dict = torch.load(os.path.join(self.oldpath, 'model' + '.pt'), map_location=self.device)
                # self.lstm.load_state_dict(model_dict)


    # def get_weights(self, edge_features):
    #     # edge_weight_list=[]
    #     B = edge_features.shape[0]
    #     E = edge_features.shape[1]
    #     edge_features = edge_features.reshape(-1,edge_features.shape[2])
    #     result = self.classifier_weights(edge_features)
    #     result = result.reshape(B,E,1)
    #     normalized_weights = F.softmax(result, dim=1).squeeze()
    #      return normalized_weights

    # def get_attention(self, outputs):
    #     #print('in attention')
    #     weights_list = []
    #     # t = time.time()
    #     for X in outputs:
    #         result = [self.attn(torch.cat((X[i], X[-1]), 0)) for i in range(X.shape[0])]
    #         result_tensor = torch.stack(result)
    #         weights_list.append(result_tensor)
    #
    #     weights = torch.stack(weights_list)
    #
    #     weights = weights.squeeze()
    #
    #     normalized_weights = F.softmax(weights, dim=1)
    #
    #     normalized_weights = normalized_weights.unsqueeze(2)
    #     attn_applied = normalized_weights * outputs
    #
    #     # attn_applied = attn_applied.squeeze()
    #     # logits = self.decoder(attn_applied)
    #     #print("attention decoder ", time.time() - t)
    #     return attn_applied

    def multi_head_attention(self, outputs):


        # t = time.time()

        keys = [self.key_layer(x) for x in outputs]
        values = [self.value_layer(x) for x in outputs]
        queries = [self.query_layer(x) for x in outputs]

        key = torch.stack(keys)
        value = torch.stack(values)
        query = torch.stack(queries)
        key = key.permute(1,0,2)
        value = value.permute(1, 0, 2)
        query = query.permute(1, 0, 2)

        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        attn_output = attn_output.permute(1,0,2)
        # print(attn_output_weights)
        return attn_output, attn_output_weights.reshape(attn_output_weights.shape[0],-1)
    def create_graphs(self, dataset, device):
        data_list = []

        for subject, label, edge_weights in dataset:
            n_nodes = subject.shape[0]
            node_features = subject.clone()
            edge_index = torch.from_numpy(np.arange(0, n_nodes)).int()
            # a = (edge_index.unsqueeze(0).repeat(n_nodes, 1))
            # b = (edge_index.unsqueeze(1).repeat(1, n_nodes))
            edge_index = torch.cat((edge_index.unsqueeze(1).repeat(1, n_nodes).reshape(1, -1),
                                    edge_index.unsqueeze(0).repeat(n_nodes, 1).reshape(1, -1)), dim=0).long()
            edge_index = edge_index.to(device)

            x = node_features

            y = label.float().view(1)

            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_weights)
            data_list.append(data)

        return data_list

    def get_encoder_loss(self, data, B):
        # data = data.permute(0,2,1,3).reshape(B,-1)
        encoder_logits = self.encoder_decoder(data.permute(0,2,1,3).reshape(B,-1))
        return encoder_logits

    # def get_lstm_loss(self, data, B):
    #     # data = data.permute(0,2,1,3).reshape(B,-1)
    #     encoder_logits = self.lstm_decoder(data.permute(0,1,2,3).reshape(B,-1))
    #     return encoder_logits

    def forward(self, sx, targets, mode='train', device="cpu",epoch=0):

        t=time.time()
        indices=""
        B = sx.shape[0]
        W = sx.shape[1]
        R = sx.shape[2]
        T = sx.shape[3]
        sx = sx.reshape(B * W, R, 1, T)
        inputs  = [self.encoder(x) for x in sx]
        #print("encoder time", time.time() - t)

        t = time.time()

        inputs = torch.stack(inputs).reshape(B, W, R, self.encoder.feature_size)
        encoder_logits = self.get_encoder_loss(inputs, B)
        inputs = inputs.permute(0,2,1,3).reshape(B,R,W*self.encoder.feature_size)
        inputs, attn_weights = self.multi_head_attention(inputs.clone())#.reshape(B,R,W,24)



        inputs = TensorDataset(inputs, targets, attn_weights)
        inputs = self.create_graphs(inputs,device)
        #
        t = time.time()
        inputs = DataLoader(inputs, batch_size=B)
        for data in inputs:
            data.batch = data.batch.to(device)
            inputs, indices = self.graph(data,epoch)
        logits = self.decoder(inputs)

        if mode == "test":
            return  encoder_logits + logits, attn_weights, indices
        else:
            return  encoder_logits + logits, "attn_weights", "indices"
