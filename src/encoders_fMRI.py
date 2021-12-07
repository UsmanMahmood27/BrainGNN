import torch
import torch.nn as nn
import torch.nn.functional as F


from .utils import init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Flatten2(nn.Module):
    def forward(self, x):
        return x.view(1, -1)

class CheckSize(nn.Module):
    def forward(self, x):
        print("final size is :   ",x.size())
        return x

class ChangeDevice(nn.Module):
    def forward(self, x, device):
        x = x.to(device)
        return x







class NatureOneCNN(nn.Module):
    def __init__(self, input_channels, args):
        super().__init__()
        self.feature_size = args.feature_size
        self.feature_size_pre_training = args.feature_size_pre_training
        self.hidden_size = self.feature_size
        self.downsample = not args.no_downsample
        self.input_channels = 1
        self.fully_connected = False
        self.end_with_relu = args.end_with_relu
        self.args = args
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.flatten = Flatten()


        self.final_conv_size = 10 * 37
        self.final_conv_shape = (10, 37)
        self.main = nn.Sequential(
            init_(nn.Conv1d(self.input_channels, 32, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv1d(32, 64, 4, stride=1)),
            nn.ReLU(),
            init_(nn.Conv1d(64, 64, 3, stride=2)),
            nn.ReLU(),
            init_(nn.Conv1d(64, 10, 1, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(self.final_conv_size, self.feature_size)),
            Flatten2(),
            init_(nn.Linear(116* self.feature_size, self.feature_size_pre_training))
        )
        # self.train()

    def forward(self, inputs, pre_training=False):

        out = self.main[0:10](inputs)
        if pre_training:
            flat = self.main[10:](out)
        else:
            flat = "junk"

        if self.end_with_relu:
            out = F.relu(out)
        else:
            return out
