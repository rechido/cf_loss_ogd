# ref: https://andrewliao11.github.io/blog/fisher-info-matrix/
# ref2: https://github.com/kuc2477/pytorch-ewc

import torch
from torch import autograd
from torch.utils.data import Dataset
import torch.nn.functional as F



# EWC (Elastic Weight Consolidation)
def compute_fisher(outputs, network):

    loglikelihoods = torch.log(outputs)
    loglikelihoods = loglikelihoods.unbind()
    loglikelihood_grads = [autograd.grad(l, network.parameters(), retain_graph=(i < len(loglikelihoods))) for i, l in enumerate(loglikelihoods, 1)]
    loglikelihood_grads = [torch.stack(gs) for gs in zip(*loglikelihood_grads)]
    fisher_diagonals = [(g ** 2).mean(0) for g in loglikelihood_grads]
    param_names = [n for n, p in network.named_parameters()]
    
    return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}



# MAS (Memory Aware Synapses)
def compute_function_sensitivity(outputs, network): # MAS

    outputs = torch.norm(outputs, 2, dim=1) ** 2
    outputs = outputs.unbind()
    outputs_grads = [autograd.grad(output, network.parameters(), retain_graph=(i < len(outputs))) for i, output in enumerate(outputs, 1)]
    outputs_grads = [torch.stack(gs) for gs in zip(*outputs_grads)]
    importance_weights = [torch.abs(g).mean(0) for g in outputs_grads]
    param_names = [n for n, p in network.named_parameters()]
    
    return {n: w.detach() for n, w in zip(param_names, importance_weights)}



class Orthonormal_Basis_Buffer(Dataset):
    def __init__(self, buffer_size, param_dim, device):
        self.ortho_basis_set = torch.zeros(buffer_size, param_dim, dtype=torch.float32, device=device)
        self.buffer_size = buffer_size
        self.length = 0

        pass

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.ortho_basis_set[idx]

    def add(self, grad_vectors):

        for vector in grad_vectors:
        
            # orthonormalization
            new_basis = vector.detach().clone().reshape(1, -1)
            prev_basis_set = self.ortho_basis_set[:self.length] # (m, p)
            projections = torch.matmul(prev_basis_set, new_basis.T) # (m, p)*(p, 1)=(m, 1) # v_dot_U
            project_vectors = projections * prev_basis_set # broadcasting (m,1) to (m,p)
            project_vector = torch.sum(project_vectors, dim=0)
            vector -= project_vector
            vector = torch.round(vector * 1e5) / 1e5 # truncate elements less than 1e-5

            self.ortho_basis_set[self.length] = F.normalize(new_basis, p=2.0, dim=0)
            self.length += 1

        pass
