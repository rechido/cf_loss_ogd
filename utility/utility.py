import torch
from torch.utils.data.dataloader import default_collate



def param_to_vector(param): # also work for grads

    flattened_param = []
    device = None

    for param_layer in param:
        if device is None:
            device = param_layer.device
        param_layer_flatten = torch.flatten(param_layer)
        flattened_param.append(param_layer_flatten)

    return torch.cat(flattened_param).to(device)



def vector_to_param(vector, network): # also works for grad (since grad has the same dimension with the parameters)
    param_sizes_flatten = []
    for param in network.parameters():
        param_flatten = torch.flatten(param)
        param_sizes_flatten.append(param_flatten.size()[0])

    param = []

    vector_split = torch.split(vector, param_sizes_flatten)
    for p, f in zip(network.parameters(), vector_split):
        param.append(f.reshape(p.size()))

    return param



def orthogonal_projection(vector, basis_set): 
    # orthogonally project vector onto basis vector set 
    # (returned vector is orthogonal to all the basis vectors)
    # basis_set must be 2d tensor
    # all basis must be unit vector
    # vector must be 1d tensor
    
    vector = vector.detach().clone().reshape(1, -1)

    projections = torch.matmul(basis_set, vector.T) # (i-1, p)*(p, 1)=(i-1, 1) # v_dot_U
    project_vectors = projections * basis_set # broadcasting
    project_vector = torch.sum(project_vectors, dim=0)
    vector -= project_vector
    vector = torch.round(vector * 1e5) / 1e5 # truncate elements less than 1e-5

    del projections, project_vectors, project_vector

    return vector.squeeze()



def compute_accuracy_matrix(task_id, test_subdatasets, accuracy_matrix, network, device, filename=None):
    network.eval()

    for j in range(len(test_subdatasets)):
        test_dataset = test_subdatasets[j]
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

        for data, target in test_loader:
            prediction = network(data)
            pred_label = torch.argmax(prediction, dim=-1)
            correct = pred_label.eq(target).sum().item()
            accuracy_matrix[task_id, j] = 100. * correct / len(target)
            break

    if filename is not None:
        with open(filename, 'w') as f:
            for row in accuracy_matrix:
                for col in row:
                    f.write('{:.2f}'.format(col))
                    f.write('\t')
                f.write('\n')