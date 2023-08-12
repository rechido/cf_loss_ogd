import numpy as np
import matplotlib.pyplot as plt
plt.style.use("default")
import argparse

from tqdm import tqdm
from datetime import date
import os


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from torchsummary import summary

from utility.visualize import plot_curve_error
from utility.utility import compute_accuracy_matrix, orthogonal_projection, compute_condition_number
from model import Model
from ogd import Orthonormal_Basis_Buffer, compute_new_basis
from dataset import make_dataset
from ewc import EWC
from replay import Episodic_Memory_Buffer, AGEM, Experience_Replay


# Setup Parameters
parser = argparse.ArgumentParser()

parser.add_argument("--dataset_path", default="~/dataset/", type=str, help="path to your dataset  ex: /hdd1/dataset/")
parser.add_argument("--dataset", default="split_mnist", type=str, choices=['split_mnist','permuted_mnist','split_cifar'])

parser.add_argument("--n_task", default=5, type=int)

parser.add_argument("--n_epoch1t", default=500, type=int)
parser.add_argument("--n_epoch", default=500, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--momentum", default=0, type=float)

parser.add_argument("--model", default="MLP", type=str, choices=['MLP','Lenet'])

# for Linear Layer
parser.add_argument("--hidden_dim", default=100, type=int)
parser.add_argument("--n_hidden_layer", default=4, type=int)

# for Conv Layer
parser.add_argument("--conv1_channel", default=20, type=int)
parser.add_argument("--conv2_channel", default=50, type=int)

parser.add_argument("--method", default="replay", type=str, choices=['sgd','ogd','pca_ogd','train_basis','ewc','agem','replay'])

# a-gem, replay
parser.add_argument("--n_examplar", default=100, type=int) # number of sample to consolidate into episodic memory buffer

# ewc
parser.add_argument("--ewc_method", default="ema", type=str, choices=['sma','ema']) # simple/exponential moving average
parser.add_argument("--fisher_sample", default=2500, type=int)
parser.add_argument("--ema_constant", default=1.0, type=float) # exponential moving average between 0~1

# ewc, replay
parser.add_argument("--regularization_constant", default=1.0, type=float)

# ogd, pca_ogd, train_basis
parser.add_argument("--n_basis", default=100, type=int) # number of sample in ogd use this instead of n_sample

# pca_ogd, train_basis
parser.add_argument("--n_sample", default=128, type=int)

# train_basis
parser.add_argument("--perturb_distance", default=0.5, type=float)
parser.add_argument("--n_epoch_b", default=5000, type=int)
parser.add_argument("--learning_rate_u", default=1e-4, type=float)
parser.add_argument("--lambda_distance", default=1e4, type=int)
parser.add_argument("--lambda_orthogonal", default=1e2, type=int)

# result folder numbering
parser.add_argument("--order", default=1, type=int)

config = parser.parse_args()

config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: {}'.format(config.device))



# Data Load

train_subdatasets, test_subdatasets = make_dataset(config)



# create result folder

today = date.today()
today = today.strftime("%Y%m%d")
config.save_folder = 'result_{}/{}_task{}_subset{}_epoch1t{}_epoch{}_batch{}_lr{}_momentum{}/' \
    .format(today, config.dataset, config.n_task, config.subset_size, config.n_epoch1t, config.n_epoch, config.batch_size, config.learning_rate, config.momentum)

if config.model in ['MLP']:
    config.save_folder += '{}_n_hidden_layer{}_hidden_dim{}/{}/' \
    .format(config.model, config.n_hidden_layer, config.hidden_dim, config.method)

if config.model in ['Lenet']:
    config.save_folder += '{}_conv1_ch{}_conv2_ch{}_n_hidden_layer{}_hidden_dim{}/{}/' \
    .format(config.model, config.conv1_channel, config.conv2_channel, config.n_hidden_layer, config.hidden_dim, config.method)


if config.method in ['ogd']:
    config.save_folder += 'basis{}/'.format(config.n_basis)

if config.method in ['pca_ogd', 'train_basis']:
    config.save_folder += 'basis{}_n_sample{}/'.format(config.n_basis, config.n_sample)

if config.method in ['train_basis']:
    config.save_folder += 'distance{}_epoch_b{}_lr_u{}_lambda_distance{}_lambda_orthogonal{}/' \
        .format(config.perturb_distance, config.n_epoch_b, config.learning_rate_u, config.lambda_distance, config.lambda_orthogonal)
    
if config.method in ['ewc']:
    if config.ewc_method in ['sma']:
        config.save_folder += 'sma_sample{}_reg{}/'.format(config.fisher_sample, config.regularization_constant)
    if config.ewc_method in ['ema']:
        config.save_folder += 'ema{}_sample{}_reg{}/'.format(config.ema_constant, config.fisher_sample, config.regularization_constant)

if config.method in ['agem']:
    config.save_folder += 'examplar{}/'.format(config.n_examplar)

if config.method in ['replay']:
    config.save_folder += 'examplar{}_reg{}/'.format(config.n_examplar, config.regularization_constant)

    

config.save_folder += '{}/'.format(config.order)

print(config.save_folder)

if not os.path.exists(config.save_folder):
    os.makedirs(config.save_folder)



# define network, loss, buffer, ...

network = Model(model_type=config.model, 
                out_dim=config.out_dim, 
                in_channel=config.in_channel, 
                hidden_dim=config.hidden_dim, 
                n_hidden_layer=config.n_hidden_layer, 
                n_head=config.n_head, 
                conv1_channel=config.conv1_channel, 
                conv2_channel=config.conv2_channel).to(config.device)
summary(network, input_size=(config.in_channel,config.img_size,config.img_size))
config.n_param = sum(p.numel() for p in network.linear.parameters())

optimizer = torch.optim.SGD(params=network.parameters(), lr=config.learning_rate, momentum=config.momentum)

criterion = nn.CrossEntropyLoss()

if config.method in ['ogd','pca_ogd','train_basis']:
    config.buffer_size = config.n_basis * (config.n_task - 1)
    buffer = Orthonormal_Basis_Buffer(config.buffer_size, config.n_param, config.device)
if config.method in ['agem']:    
    agem = AGEM()
if config.method in ['replay']:    
    er = Experience_Replay()
if config.method in ['ewc']:
    ewc = EWC(network, config)
    ewc.update_means(network)

train_losses_mean = []
train_losses_std = []

accuracy_matrix = np.zeros((config.n_task, config.n_task))
average_accuracies = np.zeros(config.n_task)

best_model = network






print('Train Start')
for task_id in range(config.n_task):
    train_dataset = train_subdatasets[task_id]    

    print("Task {}/{}".format(task_id+1, config.n_task))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, collate_fn=lambda x: tuple(x_.to(config.device) for x_ in default_collate(x)))

    if config.model in ['Lenet'] and task_id == 0:
        skip_conv = False
    else:
        skip_conv = True

    n_epoch = config.n_epoch1t if task_id == 0 else config.n_epoch

    ### Train ###
    network.train()
    for epoch in tqdm(range(n_epoch)):

        train_loss_batch = []
        for (data, labels) in train_loader:
            optimizer.zero_grad()
            # compute loss for the current task data
            task_ids = torch.full_like(labels, task_id)
            prediction = network(data, task_ids)
            train_loss = criterion(prediction, labels)
            if config.method in ['ewc'] and task_id > 0: # ewc
                train_loss = (train_loss + config.regularization_constant * ewc.penalty_loss(network)) / (1 + config.regularization_constant)
            if config.method in ['replay'] and task_id > 0: # ewc
                train_loss = (train_loss + config.regularization_constant * er.penalty_loss(network, config)) / (1 + config.regularization_constant)
            train_loss_batch.append(train_loss.item())
            train_loss.backward()

            # update network head when multihead
            if network.n_head > 1:
                param_vector = network.head_param_vector(task_id)
                grad_vector = network.head_grad_vector(task_id)
                param_vector -= config.learning_rate * grad_vector
                network.update_head(param_vector, task_id)

            # update network body
            param_vector = network.body_param_vector(skip_conv)
            grad_vector = network.body_grad_vector(skip_conv)
            if config.method in ['ogd','pca_ogd','train_basis'] and task_id > 0: # ogd
                grad_vector = orthogonal_projection(grad_vector, buffer[:len(buffer)])
            if config.method in ['agem'] and task_id > 0: # averaged-gem
                grad_vector = agem.orthogonal_projection(grad_vector, network, config, skip_conv)
            param_vector -= config.learning_rate * grad_vector # manual update
            network.update_body(param_vector, skip_conv)

        train_losses_mean.append(np.mean(train_loss_batch))
        train_losses_std.append(np.std(train_loss_batch))

    ### Train End ###

    compute_accuracy_matrix(config, task_id, test_subdatasets, accuracy_matrix, network, config.device)
    average_accuracies[task_id] = np.average(accuracy_matrix[task_id, 0 : task_id + 1])
    
    plot_curve_error(train_losses_mean, train_losses_std, 'Epoch', 'Loss', 'Train Loss', filename=config.save_folder + 'train_loss.png', show=False)  

    np.savetxt(config.save_folder + 'accuracy_matrix.txt', accuracy_matrix, fmt='%.2f', delimiter='\t')
    with open(config.save_folder + 'average_accuracy.txt', 'at') as f:
        f.write('task {}: {:.2f}\n'.format(task_id, average_accuracies[task_id]))      

    print('task {}: {}\nmean={:.2f}'.format(task_id+1, accuracy_matrix[task_id, 0 : task_id + 1], average_accuracies[task_id]))
    #print('task {}: {:.2f}'.format(task_id+1, average_accuracies[task_id]))
    
    if config.method in ['ogd','pca_ogd','train_basis'] and task_id < config.n_task - 1:
        # compute and save new basis for orthogonal projection
        new_basis = compute_new_basis(config, train_dataset, labels, network, task_id)
        buffer.add(new_basis) # save new basis with orthonormalization
    
    if config.method in ['ewc'] and task_id < config.n_task - 1: # ewc
        ewc.update_means(network) # save optimal point of model parameters
        ewc.consolidate(train_dataset, network, config, task_id) # update precision matrices

    if config.method in ['agem'] and task_id < config.n_task - 1: # save randomly sampled examplar set for replay
        agem.consolidate(train_dataset, config, task_id)

    if config.method in ['replay'] and task_id < config.n_task - 1: # save randomly sampled examplar set for replay
        er.consolidate(train_dataset, config, task_id)

    ###

    # compute the condition number of the hassian matrix
    #compute_condition_number(config, train_dataset, network, task_id)
     

print('Train End')



x = np.arange(config.n_task)

fig = plt.figure()
plt.plot(x, average_accuracies, 'o-')
plt.xlabel('Task ID')
plt.xticks(x)
plt.ylabel('Average Accuracy')
plt.show()

fig.savefig(config.save_folder + 'average_accuracy.png')

plt.close('all')
plt.cla()
plt.clf()
