import numpy as np
import matplotlib.pyplot as plt
plt.style.use("default")
import argparse

from tqdm import tqdm
from datetime import date
import os

import torch
from torch import autograd
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from utility.cl_algorithms import Orthonormal_Basis_Buffer
from utility.online_setup import make_permuted_mnist
from utility.visualize import plot_curve_error
from utility.utility import param_to_vector, vector_to_param, orthogonal_projection, compute_accuracy_matrix
from model import create_model
from ogd import compute_new_basis



# Setup Parameters
parser = argparse.ArgumentParser()

parser.add_argument("--dataset_path", default="/hdd1/dataset/", type=str, help="path to your dataset  ex: /home/usr/datasets/")
parser.add_argument("--dataset", default="permuted_mnist", type=str, help="permuted_mnist")

parser.add_argument("--n_class", default=10, type=int)
parser.add_argument("--n_task", default=2, type=int)

parser.add_argument("--subset_size_per_class", default=1000, type=int)

parser.add_argument("--n_epoch", default=1, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--momentum", default=0.0, type=float)

parser.add_argument("--model", default="MLP", type=str, help="MLP,CNN")

# for linear layer
parser.add_argument("--hidden_dim", default=100, type=int)
parser.add_argument("--n_hidden_layer", default=2, type=int)

# for conv layer
# for CNN only
parser.add_argument("--feature_dim", default=2, type=int)
parser.add_argument("--n_feature_layer", default=1, type=int)

parser.add_argument("--method", default="train_cf", type=str, help="sgd,ogd,pca_ogd,train_cf")
parser.add_argument("--n_basis", default=100, type=int)

# ogd, pca_ogd, train_cf
parser.add_argument("--n_sample", default=1000, type=int)

# train_cf
parser.add_argument("--n_epoch_cf", default=100, type=int)
parser.add_argument("--learning_rate_w", default=1e-3, type=float)
parser.add_argument("--perturb_distance", default=1, type=int)

config = parser.parse_args()

config.subset_size = config.subset_size_per_class * config.n_class
config.buffer_size = config.n_basis * (config.n_task-1)

config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: {}'.format(config.device))

today = date.today()
today = today.strftime("%Y%m%d")
config.save_folder = 'result_{}/dataset-{}_task{}_subset{}_epoch{}_batch{}_lr{}_momentum{}/model{}_hidden_dim{}_n_hidden_layer{}/{}/' \
    .format(today, config.dataset, config.n_task, config.subset_size, config.n_epoch, config.batch_size, config.learning_rate, config.momentum, config.model, config.hidden_dim, config.n_hidden_layer, config.method)

if config.method in ['ogd','pca','train_cf']:
    config.save_folder += 'basis{}_n_sample{}/'.format(config.n_basis, config.n_sample)

if config.method in ['train_cf']:
    config.save_folder += 'distance{}_epoch_cf{}_lr_w{}/'.format(config.perturb_distance, config.n_epoch_cf, config.learning_rate_w)

if not os.path.exists(config.save_folder):
    os.makedirs(config.save_folder)



# Data Load
train_dataset = MNIST(config.dataset_path, train=True, download=False,
        transform=transforms.Compose([
            transforms.Pad(2, fill=0, padding_mode='constant'),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1000,), std=(0.2752,)),
        ]))

test_dataset = MNIST(config.dataset_path, train=False, download=False,
        transform=transforms.Compose([
            transforms.Pad(2, fill=0, padding_mode='constant'),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1000,), std=(0.2752,)),
        ]))

train_subdatasets, test_subdatasets = make_permuted_mnist(config, train_dataset, test_dataset)



# define network, loss, buffer, ...

network = create_model(config)
config.n_param = sum(p.numel() for p in network.parameters())

ce_loss = nn.CrossEntropyLoss()


buffer = Orthonormal_Basis_Buffer(config.buffer_size, config.n_param, config.device)

train_losses_mean = np.zeros(config.n_epoch * config.n_task)
train_losses_std = np.zeros(config.n_epoch * config.n_task)

accuracy_matrix = np.zeros((config.n_task, config.n_task))



print('Train Start')
for task_id, train_dataset in enumerate(train_subdatasets):

    print("Task {}/{}".format(task_id+1, config.n_task))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, collate_fn=lambda x: tuple(x_.to(config.device) for x_ in default_collate(x)))

    ### Train ###
    network.train()
    for epoch in tqdm(range(config.n_epoch)):    

        train_loss_batch = []
        for (data, labels) in train_loader:
            # compute loss for the current task data
            prediction = network(data)
            train_loss = ce_loss(prediction, labels)
            train_loss_batch.append(train_loss.item())
            grad = autograd.grad(train_loss, network.parameters())

            if config.method in ['ogd','pca_ogd','train_cf'] and task_id > 0: # ogd
                gradient_vectors = param_to_vector(grad)
                new_gradient = orthogonal_projection(gradient_vectors, buffer[:config.n_basis])
                new_gradient = vector_to_param(new_gradient, network)
            else:
                new_gradient = grad

            # manually update parameters
            with torch.no_grad():
                for p, g in zip(network.parameters(), new_gradient):
                    p -= config.learning_rate * g

        train_losses_mean[config.n_epoch * task_id + epoch] = np.mean(train_loss_batch)
        train_losses_std[config.n_epoch * task_id + epoch] = np.std(train_loss_batch)

    ### Train End ###

    plot_curve_error(train_losses_mean, train_losses_std, 'Epoch', 'Loss', 'Train Loss', filename=config.save_folder + 'train_loss.png', show=False)

    compute_accuracy_matrix(task_id, test_subdatasets, accuracy_matrix, network, config.device, filename=config.save_folder + 'accuracy_matrix.txt')

    if config.method in ['ogd','pca_ogd','train_cf'] \
        and task_id < config.n_task - 1:

        n_sample = config.n_basis if config.method=='ogd' else config.n_sample

        loader = DataLoader(train_dataset, batch_size=n_sample, shuffle=True, collate_fn=lambda x: tuple(x_.to(config.device) for x_ in default_collate(x)))
        for x, y in loader:
            data = x
            label = y
            break

        new_basis = compute_new_basis(config, data, label, network, task_id)
        buffer.add(new_basis) # save new basis with orthonormalization 

    ###    

print('Train End')




