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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
import csv
import time

class Classifier(nn.Module):
    def __init__(self, num_inputs1, num_inputs2):
        super().__init__()
        self.network = nn.Bilinear(num_inputs1, num_inputs2, 1)

    def forward(self, x1, x2):
        return self.network(x1, x2)


class the_works_trainer(Trainer):
    def __init__(self, model, config, device, device_encoder, tr_labels, val_labels, test_labels, trial="", crossv="", gtrial=""):
        super().__init__(model, device)
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
        self.edge_weights= ""
        self.trials = trial
        self.gtrial = gtrial
        self.exp = config['exp']
        self.cv = crossv
        self.test_targets = ""
        self.test_predictions = ""
        self.regions_selected = ""
        self.f1=0.
        self.fpr=""
        self.tpr = ""
        self.cthreshold = ""

        self.dropout = nn.Dropout(0.65).to(self.device)

        if self.exp in ['UFPT', 'NPT']:
            self.optimizer = torch.optim.Adam(
                list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
                + list(self.model.graph.parameters())
                + list(self.model.key_layer.parameters())
                + list(self.model.value_layer.parameters()) + list(self.model.query_layer.parameters())
                +  list(self.model.multihead_attn.parameters())
                ,lr=config['lr'])

        self.early_stopper = EarlyStopping("self.model_backup",  patience=self.patience, verbose=False,
                                           wandb="self.wandb", name="model",
                                           path=self.path, trial=self.trials)
        self.transform = transforms.Compose([Cutout(n_holes=1, length=80)])



    def generate_batch(self, episodes, mode):
        if self.sample_number == 0:
            total_steps = sum([len(e) for e in episodes])
        else:
            total_steps = self.sample_number

        if mode == 'train':
            BS = self.batch_size
        else:
            BS = episodes.shape[0]
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
                # mean = episode.mean()
                # sd = episode.std()
                # episode = (episode - mean) / sd
                sx.append(episode)
            yield torch.stack(sx).to(self.device_encoder), ts_number.to(self.device_encoder)


    def do_one_epoch(self, epoch, episodes, mode):

        epoch_loss, accuracy, steps, epoch_acc, epoch_roc = 0., 0., 0, 0., 0.
        epoch_CE_loss, epoch_E_loss, epoch_lstm_loss = 0., 0., 0.,
        accuracy1, accuracy2, accuracy, FP = 0., 0., 0., 0.
        epoch_loss1, epoch_loss2, epoch_accuracy, epoch_FP = 0., 0., 0., 0.

        data_generator = self.generate_batch(episodes, mode)
        for sx, ts_number in data_generator:


            loss = 0.
            CE_loss, E_loss, lstm_loss = 0., 0., 0.

            if mode == 'train':
                targets = self.tr_labels[ts_number]

            elif mode == 'eval':
                targets = self.val_labels[ts_number]

            elif mode == 'test':
                targets = self.test_labels[ts_number]

            logits, edge_weights, regions_selected = self.model(sx, targets, mode, self.device, epoch)


            targets = targets.long()
            loss = F.cross_entropy(logits, targets)
            if mode == 'train' or mode == 'eval':
               loss, CE_loss, E_loss, lstm_loss = self.add_regularization(loss)

            #print("reg time", time.time() - t)
            t=time.time()
            accuracy, roc, pred, f1, fpr, tpr, cthreshold = self.acc_and_auc(logits.detach(), mode, targets.detach())
            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.detach().item()
            epoch_accuracy += accuracy.detach().item()


            if mode == 'train' or mode == 'eval':
                epoch_E_loss += E_loss
            if mode == 'eval' or mode == 'test':
                epoch_roc += roc
            if mode == 'test':
                self.edge_weights = edge_weights.detach()
                self.test_targets = targets.detach()
                self.test_predictions = pred
                self.regions_selected = regions_selected
                self.f1 = f1
                self.fpr = fpr
                self.tpr = tpr
                self.cthreshold = cthreshold
            del loss
            del targets
            del logits
            del pred
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
                       epoch_FP / steps, epoch_roc / steps, f1, prefix=mode)
        if mode == "eval":
            self.early_stopper(epoch_loss / steps, epoch_roc / steps, self.model, 0, epoch=epoch)
        if mode == 'test':
            self.test_accuracy = epoch_accuracy / steps
            self.test_auc = epoch_roc / steps
            self.test_loss = epoch_loss / steps
        return epoch_loss / steps

    def acc_and_auc(self, logits, mode, targets):

        sig = torch.softmax(logits, dim=1)
        values, indices = sig.max(1)
        roc = 0.
        acc = 0.
        f1=0.
        fpr =""
        tpr =""
        cthreshold=""
        y_scores = ""

        if mode == 'eval' or mode == 'test':
            if 1 in targets:
                y_scores = sig.to(self.device).detach()[:, 1]
                roc = roc_auc_score(targets.to('cpu'), y_scores.to('cpu'))
                fpr, tpr, cthreshold = roc_curve(targets.to('cpu'), y_scores.to('cpu'))
                f1 = f1_score(targets.to('cpu'),indices.to('cpu'))

        accuracy = calculate_accuracy_by_labels(indices, targets)
        if mode == 'test':
            print(indices)
            print(targets)

        return accuracy, roc, y_scores, f1, fpr, tpr, cthreshold


    def add_regularization(self, loss):
        reg = 1e-5
        E_loss = 0.
        lstm_loss = 0.
        attn_loss = 0.
        mha_loss = 0.
        CE_loss = loss

        for name, param in self.model.encoder_decoder.named_parameters():
            if 'bias' not in name:
                lstm_loss += (reg * torch.sum(torch.abs(param)))


        for name, param in self.model.graph.named_parameters():
            if 'bias' not in name:
                lstm_loss += (reg * torch.sum(torch.abs(param)))

        for name, param in self.model.key_layer.named_parameters():
            if 'bias' not in name:
                lstm_loss += (reg * torch.sum(torch.abs(param)))
        for name, param in self.model.value_layer.named_parameters():
            if 'bias' not in name:
                lstm_loss += (reg * torch.sum(torch.abs(param)))
        for name, param in self.model.query_layer.named_parameters():
            if 'bias' not in name:
                lstm_loss += (reg * torch.sum(torch.abs(param)))
        for name, param in self.model.multihead_attn.named_parameters():
            if 'bias' not in name:
                lstm_loss += (reg * torch.sum(torch.abs(param)))



        loss = loss + lstm_loss
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



        mode = 'eval'
        self.do_one_epoch(0, val_eps, mode)
        return self.test_auc

    def load_model_and_test(self, tst_eps):
        model_dict = torch.load(os.path.join(self.path, 'model' + self.trials + '.pt'), map_location=self.device)
        self.model.load_state_dict(model_dict)
        self.model.eval()



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

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=4, verbose=True )
        if self.PT in ['milc', 'milc-fMRI', 'variable-attention', 'two-loss-milc']:
            if self.exp in ['UFPT', 'FPT']:
                print('in ufpt and fpt')
                model_dict = torch.load(os.path.join(self.oldpath, 'model1' + '.pt'), map_location=self.device)
                self.model.load_state_dict(model_dict)
                self.model.to(self.device)
        self.model.init_weight(PT=self.exp)

        saved = 0
        for e in range(self.epochs):
            #print(self.exp)
            if self.exp in ['UFPT', 'NPT']:
                self.model.train()

            else:
                self.model.eval()

            mode = "train"
            val_loss = self.do_one_epoch(e, tr_eps, mode)
            self.model.eval()
            mode = "eval"
            #t = time.time()
            #print("====================================VALIDATION START===============================================")
            val_loss = self.do_one_epoch(e, val_eps, mode)
            #print("====================================VALIDATION END===============================================")

            scheduler.step(val_loss)
            if self.early_stopper.early_stop:
                self.early_stopper(0, 0, self.model, 1, epoch=e)
                saved = 1
                break

        if saved == 0:
            self.early_stopper(0, 0, self.model, 1,epoch=e)
            saved = 1

        self.save_loss_and_auc()
        self.load_model_and_test(tst_eps)


        return self.test_accuracy, self.test_auc, self.test_loss, e, self.f1, self.test_targets, self.test_predictions



    def log_results(self, epoch_idx, epoch_loss1, epoch_loss, epoch_test_accuracy, epoch_FP, epoch_roc, f1, prefix=""):
        print(
            "{} CV: {}, Trial: {}, Epoch: {}, Epoch Loss: {}, Epoch Accuracy: {}, Epoch FP: {} roc: {}, f1: {},  {}".format(
                prefix.capitalize(),
                self.cv,
                self.trials,
                epoch_idx,
                epoch_loss,
                epoch_test_accuracy,
                epoch_FP,
                epoch_roc,
                f1,
                prefix.capitalize()))
