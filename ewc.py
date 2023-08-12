from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F

class EWC(object):

    def __init__(self, network, config):
        self.ema_constant = config.ema_constant # exponential moving average between 0~1
        self.params = {n: p for n, p in network.named_parameters() if p.requires_grad}
        self.precision_matrices = {} # importance weights based on Fisher Information
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            self.precision_matrices[n] = p.data
        self.means = {} # optimal point of model parameter trained for previous tasks.
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            self.means[n] = p.data
        pass

    def consolidate(self, dataset, network, config, task_id=0):
        print("Consolidate Precision Matrices...")
        fisher_matrix = self.compute_diag_fisher(dataset, network, config, task_id)
        for index in self.precision_matrices:
            if task_id > 0:
                if config.ewc_method == 'ema':
                    self.precision_matrices[index] = self.ema_constant * fisher_matrix[index] + (1 - self.ema_constant) * self.precision_matrices[index]
                elif config.ewc_method == 'sma':
                    self.precision_matrices[index] += fisher_matrix[index]
            else:
                self.precision_matrices[index] = fisher_matrix[index]
        pass

    def update_means(self, network):
        for n, p in network.named_parameters():
            self.means[n].copy_(p.data)
        pass

    def penalty_loss(self, network):
        penalty_loss = 0        
        for n, p in network.named_parameters():
            loss = self.precision_matrices[n] * 0.5 * (p - self.means[n]) ** 2
            penalty_loss += loss.sum()
        return penalty_loss
    
    def compute_diag_fisher(self, dataset, network, config, task_id=0):
        print("Compute Fisher Information...")
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data        
        data_loader = DataLoader(dataset, batch_size=config.fisher_sample, collate_fn=lambda x: tuple(x_.to(config.device) for x_ in default_collate(x)))
        data, labels = next(iter(data_loader))
        network.eval()
        outputs = network(data, task_id)
        loss = F.cross_entropy(outputs, labels)
        network.zero_grad()
        loss.backward()
        for n, p in network.named_parameters():
            if p.grad is not None:
                precision_matrices[n].data += p.grad.data ** 2
        return precision_matrices