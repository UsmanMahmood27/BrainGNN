import random

import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from .utils import calculate_accuracy, Cutout, calculate_accuracy_by_labels, calculate_FP, calculate_FP_Max
from .newtrainer import Trainer
from src.utils import EarlyStopping
from torchvision import transforms
import matplotlib.pylab as plt
import matplotlib.pyplot as pl
import torchvision.transforms.functional as TF
import torch.nn.utils.rnn as tn
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import csv
import time

class Classifier(nn.Module):
    def __init__(self, num_inputs1, num_inputs2):
        super().__init__()
        self.network = nn.Bilinear(num_inputs1, num_inputs2, 1)

    def forward(self, x1, x2):
        return self.network(x1, x2)


class LSTMTrainer(Trainer):
    def __init__(self, model, config, device, device_encoder, tr_labels, val_labels, test_labels, trial="", crossv="", gtrial=""):
        super().__init__(model , device)
        self.config = config
        self.device_encoder = device_encoder
        self.tr_labels = tr_labels
        self.test_labels = test_labels
        self.val_labels = val_labels
        self.patience = self.config["patience"]
        self.dropout = nn.Dropout(0.65).to(device)
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.sample_number = config['sample_number']
        self.path = config['path']
        self.oldpath = config['oldpath']
        self.fig_path = config['fig_path']
        self.p_path = config['p_path']
        self.PT = config['pre_training']
        self.device = device
        self.gain = config['gain']
        self.train_epoch_loss, self.train_batch_loss, self.eval_epoch_loss, self.eval_batch_loss, self.eval_batch_accuracy, self.train_epoch_accuracy = [], [], [], [], [], []
        self.train_epoch_roc, self.eval_epoch_roc = [], []
        self.eval_epoch_CE_loss, self.eval_epoch_E_loss, self.eval_epoch_lstm_loss = [], [], []
        self.test_accuracy = 0.
        self.test_auc = 0.
        self.test_loss = 0.
        self.n_heads = 1
        self.trials = trial
        self.gtrial = gtrial
        self.exp = config['exp']
        self.cv = crossv
        self.attn = nn.Sequential(
            nn.Linear(2 * self.lstm.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128,1)
        ).to(device)

        self.final_attn = nn.Linear(140 * 2, 2).to(device)

        self.key_layer = nn.Sequential(
            nn.Linear(self.lstm.hidden_dim, 2),
        ).to(device)

        self.value_layer = nn.Sequential(
            nn.Linear(self.lstm.hidden_dim, 2),
        ).to(device)

        self.query_layer = nn.Sequential(
            nn.Linear(self.lstm.hidden_dim, 2),
        ).to(device)

        self.multihead_attn = nn.MultiheadAttention(2, self.n_heads).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(self.lstm.hidden_dim, self.lstm.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.lstm.hidden_dim, 2),

        ).to(device)
        self.init_weight()
        self.dropout = nn.Dropout(0.65).to(self.device)

        if self.exp in ['UFPT', 'NPT']:
            self.optimizer = torch.optim.Adam(
                list(self.decoder.parameters()) + list(self.lstm.parameters()) + list(
                    self.encoder.parameters()) + list(self.attn.parameters()),lr=config['lr'], eps=1e-5)
        else:
            if self.PT in ['milc', 'milc-fMRI', 'variable-attention', 'two-loss-milc']:
                self.optimizer = torch.optim.Adam(list(self.decoder.parameters()),lr=config['lr'], eps=1e-5)
            else:
                self.optimizer = torch.optim.Adam(list(self.decoder.parameters()) + list(self.attn.parameters())
                                                       + list(self.lstm.parameters()) + list(self.key_layer.parameters())
                                                  + list(self.value_layer.parameters()) + list(self.query_layer.parameters())
                                                  +  list(self.multihead_attn.parameters()),
                                                  lr=config['lr'], eps=1e-5)

        self.encoder_backup = self.encoder
        self.lstm_backup = self.lstm
        self.early_stopper = EarlyStopping(self.encoder_backup, self.lstm_backup, patience=self.patience, verbose=False,
                                           wandb=self.wandb, name="encoder",
                                           path=self.path, trial=self.trials)
        self.transform = transforms.Compose([Cutout(n_holes=1, length=80)])

    def init_weight(self):
        for name, param in self.decoder.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.attn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)

    def generate_batch(self, episodes, mode):
        if self.sample_number == 0:
            total_steps = sum([len(e) for e in episodes])
        else:
            total_steps = self.sample_number
        # print('Total Steps: {}'.format(total_steps))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        if mode == 'train':
            BS = self.batch_size
        else:
            BS = self.batch_size
        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=False),
                               BS, drop_last=False)

        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]
            ts_number = torch.LongTensor(indices)
            i = 0
            sx = []
            for episode in episodes_batch:
                # Get all samples from this episode
                mean = episode.mean()
                sd = episode.std()
                episode = (episode - mean) / sd
                sx.append(episode)
            yield torch.stack(sx).to(self.device_encoder), ts_number.to(self.device_encoder)


    def do_one_epoch(self, epoch, episodes, mode):
        #print('lstm device epoch start  ' + torch.cuda.get_device_name(device=self.device) + ' ', (torch.cuda.memory_allocated(device=self.device) / (1024*1024)),  (torch.cuda.memory_cached(device=self.device) / (1024*1024)))
        #print('encoder device epoch start  '  + torch.cuda.get_device_name(device=self.device_encoder) + ' ', (torch.cuda.memory_allocated(device=self.device_encoder) / (1024*1024)),  (torch.cuda.memory_cached(device=self.device_encoder) / (1024*1024)) )
        #torch.cuda.empty_cache()
        # mode = "train" if self.encoder.training and self.classifier1.training else "val"
        epoch_loss, accuracy, steps, epoch_acc, epoch_roc = 0., 0., 0, 0., 0.
        epoch_CE_loss, epoch_E_loss, epoch_lstm_loss = 0., 0., 0.,
        accuracy1, accuracy2, accuracy, FP = 0., 0., 0., 0.
        epoch_loss1, epoch_loss2, epoch_accuracy, epoch_FP = 0., 0., 0., 0.

        data_generator = self.generate_batch(episodes, mode)
        for sx, ts_number in data_generator:
            #torch.cuda.empty_cache()
            #print('lstm device for start  ' + torch.cuda.get_device_name(device=self.device) + ' ', (torch.cuda.memory_allocated(device=self.device) / (1024*1024)),  (torch.cuda.memory_cached(device=self.device) / (1024*1024)))
            #print('encoder device for start  '  + torch.cuda.get_device_name(device=self.device_encoder) + ' ', (torch.cuda.memory_allocated(device=self.device_encoder) / (1024*1024)),  (torch.cuda.memory_cached(device=self.device_encoder) / (1024*1024)))
            loss = 0.
            CE_loss, E_loss, lstm_loss = 0., 0., 0.


            # print('sx', sx.device, sx.dtype, type(sx), sx.type())
            #t=time.time()
            # sx = sx.reshape(sx.shape[0],sx.shape[1],1,sx.shape[2],sx.shape[3])
            #comb = [x for x in sx]
            s1 = sx.shape[0]
            s2 = sx.shape[1]
            #print(s1)
            #print(s2)
            #sx = sx.reshape(sx.shape[0] * sx.shape[1], sx.shape[2], sx.shape[3], sx.shape[4])
            #print(sx.shape)
            #inputs = self.encoder(sx, device=self.device)
            #print(inputs.shape)

            inputs = [self.encoder(x, device=self.device) for x in sx]
            #print(len(inputs))
            #inputs = inputs.reshape(s1,s2,inputs.shape[1])

            #print("inputs time",time.time()-t)
            #t=time.time()
            outputs = self.lstm(torch.stack(inputs), mode)
            #print("ouputs time", time.time() - t)
            #t=time.time()
            # logits = self.get_attention(outputs)
            logits = self.multi_head_attention()
            #logits = logits.to(self.device)
            #print("attention time", time.time() - t)
            #t = time.time()
            if mode == 'train':
                targets = self.tr_labels[ts_number]

            elif mode == 'eval':
                targets = self.val_labels[ts_number]

            elif mode == 'test':
                targets = self.test_labels[ts_number]

            targets = targets.long()
            loss = F.cross_entropy(logits, targets)
            #print("loss time", time.time() - t)
            # regularization
            #t=time.time()
            if mode == 'train' or mode == 'eval':
               loss, CE_loss, E_loss, lstm_loss = self.add_regularization(loss)

            #print("reg time", time.time() - t)
            t=time.time()
            accuracy, roc = self.acc_and_auc(logits.detach(), mode, targets.detach())
            #print("auc time", time.time() - t)
            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            #t = time.time()

            epoch_loss += loss.detach().item()
            epoch_accuracy += accuracy.detach().item()
            del loss
            del targets
            del logits
            del outputs
            del inputs
            if mode == 'train' or mode == 'eval':
                # epoch_CE_loss += CE_loss.detach().item()
                epoch_E_loss += E_loss
                # epoch_lstm_loss += lstm_loss.detach().item()
            if mode == 'eval' or mode == 'test':
                epoch_roc += roc
            #print("junk time", time.time() - t)
            steps += 1

        #t = time.time()
        if mode == "eval":
            self.eval_batch_accuracy.append(epoch_accuracy / steps)
            self.eval_epoch_loss.append(epoch_loss / steps)
            self.eval_epoch_roc.append(epoch_roc / steps)
            self.eval_epoch_CE_loss.append(epoch_CE_loss / steps)
            self.eval_epoch_E_loss.append(epoch_E_loss / steps)
            self.eval_epoch_lstm_loss.append(epoch_lstm_loss / steps)
        elif mode == 'train':
            self.train_epoch_loss.append(epoch_loss / steps)
            self.train_epoch_accuracy.append(epoch_accuracy / steps)
        if epoch % 1 == 0:
          self.log_results(epoch, epoch_loss1 / steps, epoch_loss / steps, epoch_accuracy / steps,
                       epoch_FP / steps, epoch_roc / steps, prefix=mode)
        if mode == "eval":
            self.early_stopper(epoch_loss / steps, epoch_roc / steps, self.encoder, self.lstm, self.attn, self.decoder,
                               0, self.key_layer, self.value_layer, self.query_layer, self.multihead_attn)
        if mode == 'test':
            self.test_accuracy = epoch_accuracy / steps
            self.test_auc = epoch_roc / steps
            self.test_loss = epoch_loss / steps
        #print("last time", time.time() - t)
        return epoch_loss / steps

    def acc_and_auc(self, logits, mode, targets):
        # N = logits.size(0)
        # sig = torch.zeros(N, 2).to(self.device)
        sig = torch.softmax(logits, dim=1)
        values, indices = sig.max(1)
        roc = 0.
        acc = 0.
        # y_scores = sig.detach().gather(1, targets.to(self.device).long().view(-1,1))
        if mode == 'eval' or mode == 'test':
            y_scores = sig.to(self.device).detach()[:, 1]
            roc = roc_auc_score(targets.to('cpu'), y_scores.to('cpu'))
        accuracy = calculate_accuracy_by_labels(indices, targets)

        return accuracy, roc

    def multi_head_attention(self, outputs):


        # t = time.time()

        keys = [self.key_layer(x) for x in outputs]
        values = [self.value_layer(x) for x in outputs]
        queries = [self.query_layer(x) for x in outputs]

        key = torch.stack(keys).permute(1, 0, 2)
        value = torch.stack(values).permute(1, 0, 2)
        query = torch.stack(queries).permute(1, 0, 2)


        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        attn_output = attn_output.permute(1, 0, 2)
        #attn_output = attn_output.sum(dim=1)
        attn_output = attn_output.reshape(attn_output.shape[0], attn_output.shape[1] * attn_output.shape[2])
        attn_output = self.final_attn(attn_output)
        logits=self.decoder(attn_output)
        #logits = self.get_attention(attn_output)


        return logits



    def get_attention(self, outputs):
        # print('Outputs From LSTM:', outputs.shape)

        #weights_list = []
        #t = time.time()
        #for X in outputs:
        #    #t=time.time()
        #    result = [torch.cat((X[i], X[-1]), 0) for i in range(X.shape[0])]
        #    result = torch.stack(result)
        #    result_tensor = self.attn(result)
        #    weights_list.append(result_tensor)
        #print("attention loop time 1 ", time.time() - t)

        weights_list = []
        #t = time.time()
        for X in outputs:
            result = [self.attn(torch.cat((X[i], X[-1]), 0)) for i in range(X.shape[0])]
            result_tensor = torch.stack(result)
            weights_list.append(result_tensor)
        #print("attention loop time 2 ", time.time()-t)

        #t = time.time()
        weights = torch.stack(weights_list)
        #print("weights stack ", time.time() - t)
        #t = time.time()
        weights = weights.squeeze()
        #print("weights squeeze ", time.time() - t)
        # print('Weights Shape:', weights.shape)

        # SoftMax normalization on weights
        #t = time.time()
        normalized_weights = F.softmax(weights, dim=1)
        #print("attention normalized weights ", time.time() - t)
        # Batch-wise multiplication of weights and lstm outputs
        #t = time.time()

        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)
        #print("attention bmm ", time.time() - t)
        #t=time.time()
        attn_applied = attn_applied.squeeze()
        #print("attention final squeeze ", time.time() - t)
        # print('After attention shape:', attn_applied.shape)

        # Pass the weighted output to decoder
        #t = time.time()
        logits = self.decoder(attn_applied)
        #print("attention decoder ", time.time() - t)
        return logits

    def add_regularization(self, loss):
        reg = 1e-3
        E_loss = 0.
        lstm_loss = 0.
        attn_loss = 0.
        mha_loss = 0.
        CE_loss = loss
        # for name, param in self.encoder.named_parameters():
        #     if 'bias' not in name:
        #         E_loss += (reg * torch.sum(torch.abs(param)))

        for name, param in self.lstm.named_parameters():
            if 'bias' not in name:
                lstm_loss += (reg * torch.sum(torch.abs(param)))
        #
        for name, param in self.attn.named_parameters():
            if 'bias' not in name:
                attn_loss += (reg * torch.sum(torch.abs(param)))

        # for name, param in self.decoder.named_parameters():
        #     if 'bias' not in name:
        #         attn_loss += (reg * torch.sum(torch.abs(param)))


        #for name, param in self.multihead_attn.named_parameters():
        #    if 'bias' not in name:
        #        mha_loss += (reg * torch.sum(torch.abs(param)))

        loss = loss + lstm_loss + attn_loss
        return loss, CE_loss, E_loss, lstm_loss

    def validate(self, val_eps):

        model_dict = torch.load(os.path.join(self.p_path, 'encoder' + self.trials + '.pt'), map_location=self.device)
        self.encoder.load_state_dict(model_dict)
        self.encoder.eval()
        self.encoder.to(self.device)

        model_dict = torch.load(os.path.join(self.p_path, 'lstm' + self.trials + '.pt'), map_location=self.device)
        self.lstm.load_state_dict(model_dict)
        self.lstm.eval()
        self.lstm.to(self.device)

        # model_dict = torch.load(os.path.join(self.p_path, 'decoder' + self.trials + '.pt'), map_location=self.device)
        # self.decoder.load_state_dict(model_dict)
        # self.decoder.eval()
        # self.decoder.to(self.device)

        mode = 'eval'
        self.do_one_epoch(0, val_eps, mode)
        return self.test_auc

    def load_model_and_test(self, tst_eps):
        model_dict = torch.load(os.path.join(self.path, 'encoder' + self.trials + '.pt'), map_location=self.device)
        self.encoder.load_state_dict(model_dict)
        self.encoder.eval()
        #self.encoder.to(self.device_encoder)

        model_dict = torch.load(os.path.join(self.path, 'lstm' +  self.trials + '.pt'), map_location=self.device)
        self.lstm.load_state_dict(model_dict)
        self.lstm.eval()
        self.lstm.to(self.device)

        model_dict = torch.load(os.path.join(self.path, 'attn' +  self.trials + '.pt'), map_location=self.device)
        self.attn.load_state_dict(model_dict)
        self.attn.eval()
        self.attn.to(self.device)

        model_dict = torch.load(os.path.join(self.path, 'cone' + self.trials + '.pt'), map_location=self.device)
        self.decoder.load_state_dict(model_dict)
        self.decoder.eval()
        self.decoder.to(self.device)

        #model_dict = torch.load(os.path.join(self.path, 'key' + self.trials + '.pt'), map_location=self.device)
        #self.key_layer.load_state_dict(model_dict)
        #self.key_layer.eval()
        #self.key_layer.to(self.device)

        #model_dict = torch.load(os.path.join(self.path, 'value' + self.trials + '.pt'), map_location=self.device)
        #self.value_layer.load_state_dict(model_dict)
        #self.value_layer.eval()
        #self.value_layer.to(self.device)

        #model_dict = torch.load(os.path.join(self.path, 'query' + self.trials + '.pt'), map_location=self.device)
        #self.query_layer.load_state_dict(model_dict)
        #self.query_layer.eval()
        #self.query_layer.to(self.device)

        #model_dict = torch.load(os.path.join(self.path, 'mha' + self.trials + '.pt'), map_location=self.device)
        #self.multihead_attn.load_state_dict(model_dict)
        #self.multihead_attn.eval()
        #self.multihead_attn.to(self.device)


        mode = 'test'
        self.do_one_epoch(0, tst_eps, mode)

    def save_loss_and_auc(self):

        with open(os.path.join(self.path, 'all_data_information' + self.trials + '.csv'), 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            self.train_epoch_loss.insert(0, 'train_epoch_loss')
            wr.writerow(self.train_epoch_loss)

            self.train_epoch_accuracy.insert(0, 'train_epoch_accuracy')
            wr.writerow(self.train_epoch_accuracy)

            self.eval_epoch_loss.insert(0, 'eval_epoch_loss')
            wr.writerow(self.eval_epoch_loss)

            self.eval_batch_accuracy.insert(0, 'eval_batch_accuracy')
            wr.writerow(self.eval_batch_accuracy)

            self.eval_epoch_roc.insert(0, 'eval_epoch_roc')
            wr.writerow(self.eval_epoch_roc)

            self.eval_epoch_CE_loss.insert(0, 'eval_epoch_CE_loss')
            wr.writerow(self.eval_epoch_CE_loss)

            self.eval_epoch_E_loss.insert(0, 'eval_epoch_E_loss')
            wr.writerow(self.eval_epoch_E_loss)

            self.eval_epoch_lstm_loss.insert(0, 'eval_epoch_lstm_loss')
            wr.writerow(self.eval_epoch_lstm_loss)

    def train(self, tr_eps, val_eps, tst_eps):
        # TODO: Make it work for all modes, right now only it defaults to pcl.
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[4, 30, 128, 256, 512, 700, 800, 2500], gamma=0.15)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min' )
        if self.PT in ['milc', 'milc-fMRI', 'variable-attention', 'two-loss-milc']:
            if self.exp in ['UFPT', 'FPT']:
                print('in ufpt and fpt')
                model_dict = torch.load(os.path.join(self.oldpath, 'lstm' + '.pt'), map_location=self.device)
                self.lstm.load_state_dict(model_dict)
                self.lstm.to(self.device)

                model_dict = torch.load(os.path.join(self.oldpath, 'attn' + '.pt'), map_location=self.device)
                self.attn.load_state_dict(model_dict)
                self.attn.to(self.device)

                #model_dict = torch.load(os.path.join(self.oldpath, 'key' + '.pt'), map_location=self.device)
                #self.key_layer.load_state_dict(model_dict)
                #self.key_layer.to(self.device)

                #model_dict = torch.load(os.path.join(self.oldpath, 'value' + '.pt'), map_location=self.device)
                #self.value_layer.load_state_dict(model_dict)
                #self.value_layer.to(self.device)

                #model_dict = torch.load(os.path.join(self.oldpath, 'query' + '.pt'), map_location=self.device)
                #self.query_layer.load_state_dict(model_dict)
                #self.query_layer.to(self.device)

                #model_dict = torch.load(os.path.join(self.oldpath, 'mha' + '.pt'), map_location=self.device)
                #self.multihead_attn.load_state_dict(model_dict)
                #self.multihead_attn.to(self.device)
        saved = 0
        for e in range(self.epochs):
            #print(self.exp)
            if self.exp in ['UFPT', 'NPT']:
                self.encoder.train(), self.lstm.train(), self.attn.train(),
                self.key_layer.train(), self.value_layer.train(), self.query_layer.train(), self.multihead_attn.train()

            else:
                self.encoder.eval(), self.lstm.train(), self.attn.train()
                self.key_layer.train(), self.value_layer.train(), self.query_layer.train(), self.multihead_attn.eval()

            mode = "train"
            #t = time.time()
            val_loss = self.do_one_epoch(e, tr_eps, mode)
            #print("train time", time.time()-t)
            self.encoder.eval(), self.lstm.eval(), self.attn.eval(), self.multihead_attn.eval()
            mode = "eval"
            #t = time.time()
            #print("====================================VALIDATION START===============================================")
            val_loss = self.do_one_epoch(e, val_eps, mode)
            #print("val time",time.time()-t)
            #print("====================================VALIDATION END===============================================")
            scheduler.step(val_loss)
            if self.early_stopper.early_stop:
                self.early_stopper(0, 0, self.encoder, self.lstm, self.attn, self.decoder, 1,
                                   self.key_layer, self.value_layer, self.query_layer, self.multihead_attn)
                saved = 1
                break

        if saved == 0:
            self.early_stopper(0, 0, self.encoder, self.lstm, self.attn, self.decoder, 1,
                               self.key_layer, self.value_layer, self.query_layer, self.multihead_attn)
            saved = 1

        self.save_loss_and_auc()
        self.load_model_and_test(tst_eps)

        # f = pl.figure()
        #
        # pl.plot(self.train_epoch_loss[1:], label='train_total_loss')
        # pl.plot(self.eval_epoch_loss[1:], label='val_total_loss')
        # pl.plot(self.eval_epoch_CE_loss[1:], label='val_CE_loss')
        # pl.plot(self.eval_epoch_E_loss[1:], label='val_Enc_loss')
        # pl.plot(self.eval_epoch_lstm_loss[1:], label='val_lstm_loss')
        # # #
        # #
        # pl.xlabel('epochs')
        # pl.ylabel('loss')
        # pl.legend()
        # pl.show()
        # f.savefig(os.path.join(self.fig_path, 'all_loss.png'), bbox_inches='tight')
        #
        # f = pl.figure()
        # #
        # pl.plot(self.train_epoch_accuracy[1:], label='train_acc')
        # pl.plot(self.eval_batch_accuracy[1:], label='val_acc')
        # pl.plot(self.eval_epoch_roc[1:], label='val_auc')
        #
        # #
        #
        # pl.xlabel('epochs')
        # pl.ylabel('acc/auc')
        # pl.legend()
        # pl.show()
        # f.savefig(os.path.join(self.fig_path, 'acc.png'), bbox_inches='tight')

        return self.test_accuracy, self.test_auc, self.test_loss, e
        # return self.early_stopper.val_acc_max



    def log_results(self, epoch_idx, epoch_loss1, epoch_loss, epoch_test_accuracy, epoch_FP, epoch_roc, prefix=""):
        print(
            "{} CV: {}, Trial: {}, Gtrial: {}, Epoch: {}, Epoch Loss: {}, Epoch Accuracy: {}, Epoch FP: {} roc: {},  {}".format(
                prefix.capitalize(),
                self.cv,
                self.trials,
                self.gtrial,
                epoch_idx,
                epoch_loss,
                epoch_test_accuracy,
                epoch_FP,
                epoch_roc,
                prefix.capitalize()))
