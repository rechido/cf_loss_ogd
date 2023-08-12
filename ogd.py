import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate


from tqdm import tqdm
import copy

from utility.utility import compute_prediction_gradients, vector_to_param, check_orthogonality
from utility.visualize import plot_curve_error



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
            new_basis -= project_vector

            if torch.norm(new_basis, p=2.0, dim=-1) > 1e-3:
                self.ortho_basis_set[self.length] = F.normalize(new_basis.squeeze(), p=2.0, dim=-1)
                self.length += 1

        pass



def compute_new_basis(config, train_dataset, labels, network, task_id):

    if config.method == 'ogd':
        new_basis = ogd(config, train_dataset, labels, network, task_id)

    elif config.method == 'pca_ogd':
        new_basis = pca_ogd(config, train_dataset, labels, network, task_id)

        check_orthogonality(new_basis, filename=config.save_folder + "orthogonality_check.txt")

    elif config.method == 'train_basis':
        new_basis = train_basis(config, train_dataset, network, task_id)

    else:
        assert False, "Invalid method_type"

    return new_basis



def ogd(config, train_dataset, label, network, task_id):

    loader = DataLoader(train_dataset, batch_size=config.n_basis, shuffle=True, collate_fn=lambda x: tuple(x_.to(config.device) for x_ in default_collate(x)))
    data, label = next(iter(loader))

    new_basis = compute_prediction_gradients(config, data, label, network, task_id)

    return new_basis



def pca_ogd(config, train_dataset, label, network, task_id):

    loader = DataLoader(train_dataset, batch_size=config.n_sample, shuffle=True, collate_fn=lambda x: tuple(x_.to(config.device) for x_ in default_collate(x)))
    data, label = next(iter(loader))

    new_basis = compute_prediction_gradients(config, data, label, network, task_id)

    print('running SVD...')
    _, _, eigenvectors_t = torch.svd_lowrank(new_basis, q=config.n_basis)

    new_basis = eigenvectors_t.T

    return new_basis



def train_basis(config, train_dataset, network, task_id):

    print('compute and save basis set for orthogonal projection...')
    print("Train Maximum Catastrophic Forgetting Direction: Panelty")    
    
    mse_loss = nn.MSELoss()

    U = torch.randn(config.n_basis, config.n_param, device=config.device)
    U = F.normalize(U, p=2.0, dim=1) * config.perturb_distance

    U.requires_grad_()
    optimizer = torch.optim.Adam([U], lr=config.learning_rate_u)

    new_basis = None
    max_cf = 0

    cf_losses = []
    distance_losses = []
    orthogonal_losses = []
    total_losses = []
    for epoch_b in tqdm(range(config.n_epoch_b)):

        data_loader = DataLoader(train_dataset, batch_size=config.n_sample, shuffle=True, drop_last=True, collate_fn=lambda x: tuple(x_.to(config.device) for x_ in default_collate(x)))

        for data, labels in data_loader:

            optimizer.zero_grad()

            cf_loss = 0

            for u in U:

                network_perturb = copy.deepcopy(network)
                u_grad = vector_to_param(u, network_perturb)
                
                for p, g in zip(network_perturb.linear.parameters(), u_grad): # initialize perturbed model parameters                
                    g = g.to(config.device)
                    p.detach_()
                    p.copy_(p + g)

                prediction = network(data, task_id)
                prediction_perturb = network_perturb(data, task_id)
                cf_loss += -mse_loss(prediction_perturb, prediction) / config.n_basis # maximize difference between prediction_modified and prediction

            cf_losses.append(cf_loss.item())

            cf = -cf_loss.item() # catastrophic forgetting
            if cf > max_cf:
                max_cf = cf
                new_basis = copy.deepcopy(U)

            # distance constraint
            U_norm = torch.norm(U, p=2.0, dim=1)
            distance_target = torch.full_like(U_norm, config.perturb_distance)
            distance_loss = mse_loss(U_norm, distance_target)
            distance_losses.append(distance_loss.item())

            # orthogonality constraint
            U_normalized = F.normalize(U, p=2.0, dim=1)
            u_dot_u = torch.matmul(U_normalized, U_normalized.T) # (d, d)
            u_dot_u_sq = u_dot_u ** 2
            u_dot_u_f_norm_sq = torch.sum(u_dot_u_sq)
            orthogonal_loss = 0.5 * (u_dot_u_f_norm_sq - torch.trace(u_dot_u_sq))
            orthogonal_losses.append(orthogonal_loss.item())

            total_loss = (cf_loss + config.lambda_distance * distance_loss + config.lambda_orthogonal * orthogonal_loss) / (1 + config.lambda_distance + config.lambda_orthogonal)
            total_losses.append(total_loss.item())

            total_loss.backward()
            optimizer.step()

            break

        #with open(config.save_folder+'cf_loss_{}.txt'.format(task_id), 'at') as f:
        #    f.write('{:.6f}\n'.format(cf_loss.item()))

    loss_log = 'task {}: cf_loss={:.6f}, distance_loss={:.6f}, orthogonal_loss={:.6f}, total_loss={:.6f}, max_cf={:.6f}\n'.format(task_id+1, cf_losses[-1], distance_losses[-1], orthogonal_losses[-1], total_losses[-1], max_cf)
    print(loss_log)
    with open(config.save_folder+'final_losses.txt', 'at') as f:
        f.write(loss_log)

    plot_curve_error(cf_losses, None, 'Iteration', 'Loss', 'Catastrophic Forgetting Loss', filename=config.save_folder + 'cf_loss_{}.png'.format(task_id), show=False)
    plot_curve_error(distance_losses, None, 'Iteration', 'Loss', 'Distance Constraint Loss', filename=config.save_folder + 'distance_loss_{}.png'.format(task_id), show=False)
    plot_curve_error(orthogonal_losses, None, 'Iteration', 'Loss', 'Orthogonality Constraint Loss', filename=config.save_folder + 'orthogonal_loss_{}.png'.format(task_id), show=False)
    plot_curve_error(total_losses, None, 'Iteration', 'Loss', 'Total Loss', filename=config.save_folder + 'total_loss_{}.png'.format(task_id), show=False)

    new_basis.detach_()
    new_basis = F.normalize(new_basis, p=2.0, dim=1)

    return new_basis