import torch

class Trainer():
    def __init__(self, model, device=torch.device('cpu')):
        self.model = model
        self.device = device

    def generate_batch(self, episodes, mode):
        raise NotImplementedError

    def train(self, episodes):
        raise NotImplementedError

    def log_results(self, epoch_idx, epoch_loss, accuracy):
        raise NotImplementedError