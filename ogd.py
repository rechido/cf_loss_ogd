import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F

from tqdm import tqdm
import copy

from utility.utility import param_to_vector, vector_to_param, orthogonal_projection
from utility.visualize import plot_curve_error



def compute_new_basis(config, data, label, network, task_id):

    if config.method == 'ogd':
        new_basis = ogd(config, data, label, network)

    elif config.method == 'pca_ogd':
        new_basis = pca_ogd(config, data, label, network)

    elif config.method == 'train_cf':
        new_basis = train_cf(config, data, network, task_id)

    return new_basis




def compute_prediction_gradients(config, data, label, network):

    prediction = network(data)
    pred_gtl = torch.zeros(len(prediction))
    for i in range(len(prediction)):
        pred_gtl[i] = prediction[i, label[i]] # OGD-GTL

    print('compute the gradients of the prediction for the present task')
    gradient_vectors = torch.zeros(len(pred_gtl), config.n_param, device=config.device)
    for n, p in enumerate(tqdm(pred_gtl)):
        grad = autograd.grad(p, network.parameters(), retain_graph=True)
        gradient_vectors[n] = param_to_vector(grad)
    
    return gradient_vectors



def ogd(config, data, label, network):

    new_basis = compute_prediction_gradients(config, data, label, network)

    return new_basis



def pca_ogd(config, data, label, network):

    gradient_vectors = compute_prediction_gradients(config, data, label, network)

    print('running SVD...')
    _, _, eigenvectors_t = torch.svd_lowrank(gradient_vectors, q=config.n_basis)

    new_basis = eigenvectors_t.T

    return new_basis



def train_cf(config, data, network, task_id):

    print("Train Maximum Catastrophic Forgetting Direction")
    print('compute and save basis set for orthogonal projection')

    new_basis = torch.zeros(config.n_basis, config.n_param, device=config.device)

    mse_loss = nn.MSELoss()

    cf_losses = []
    perturb_distances = []
    max_cfs = []
    for n in tqdm(range(config.n_basis)):

        network_perturb = copy.deepcopy(network)
        network_perturb.train()

        perturb_vector = torch.randn(config.n_param, device=config.device)
        if n > 0:
            prev_grad_vectors = new_basis[:n]
            perturb_vector = orthogonal_projection(perturb_vector, prev_grad_vectors)
        perturb_vector = F.normalize(perturb_vector, p=2.0, dim=0) * config.perturb_distance

        perturb_grad = vector_to_param(perturb_vector, network_perturb)
        with torch.no_grad():
            for p, g in zip(network_perturb.parameters(), perturb_grad): # initialize perturbed model parameters
                p += g

        max_cf = 0
        for epoch_cf in range(config.n_epoch_cf):

            prediction = network(data)
            prediction_perturb = network_perturb(data)
            cf_loss = -mse_loss(prediction_perturb, prediction) # maximize difference between prediction_modified and prediction
            cf_losses.append(cf_loss.item())
            if -cf_loss.item() > max_cf:
                max_cf = -cf_loss.item()

            # distance constraint
            w_perturb = param_to_vector(network_perturb.parameters())
            w = param_to_vector(network.parameters())
            u = (w_perturb - w).detach()
            d_perturb = torch.norm(u, p=2.0, dim=0)
            perturb_distances.append(d_perturb.item())
            grad = autograd.grad(cf_loss, network_perturb.parameters())
            grad_vector = param_to_vector(grad)

            # remove diverging direction from the gradient
            grad_vector -= torch.dot(grad_vector, u) / torch.dot(u, u) * u # orthogonal projection against u

            if n > 0: # ogd            
                grad_vector = orthogonal_projection(grad_vector, prev_grad_vectors)
            
            new_gradient = vector_to_param(grad_vector, network_perturb)

            # manually update parameters
            with torch.no_grad():
                for p, g in zip(network_perturb.parameters(), new_gradient):
                    p -= config.learning_rate_w * g

            # projection to the perturb_distance
            w_perturb = param_to_vector(network_perturb.parameters())
            w = param_to_vector(network.parameters())
            u = (w_perturb - w).detach()
            d_perturb = torch.norm(u, p=2.0, dim=0)
            if d_perturb > config.perturb_distance:
                u_renormalized = F.normalize(u, p=2.0, dim=0) * config.perturb_distance
                w_perturb = w + u_renormalized
                w_perturb_param = vector_to_param(w_perturb, network_perturb)
                with torch.no_grad():
                    for p, w in zip(network_perturb.parameters(), w_perturb_param):
                        p.copy_(w)

            #print('basis: {}, epoch: {}, cf_loss: {}'.format(n, epoch_cf, cf_losses[-1]))

        w_perturb = param_to_vector(network_perturb.parameters())
        w = param_to_vector(network.parameters())
        new_basis[n] = (w_perturb - w).detach()
        max_cfs.append(max_cf)

    plot_curve_error(cf_losses, None, 'Iteration', 'Loss', 'Catastrophic Forgetting Loss', filename=config.save_folder + 'cf_loss_{}.png'.format(task_id), show=False)
    plot_curve_error(perturb_distances, None, 'Iteration', 'Distance', 'Perturbation Distance', filename=config.save_folder + 'perturb_distances_{}.png'.format(task_id), show=False)
    plot_curve_error(max_cfs, None, 'Basis', 'CF Loss', 'Maximized CF Loss per each Basis', filename=config.save_folder + 'max_cfs_{}.png'.format(task_id), show=False)

    return new_basis