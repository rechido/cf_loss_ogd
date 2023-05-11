from tqdm import tqdm

import torch
from torch.utils.data.dataloader import default_collate



def vector_to_param(vector, network): # also works for grad (since grad has the same dimension with the parameters)
    param_sizes_flatten = []
    for param in network.linear.parameters():
        param_flatten = torch.flatten(param)
        param_sizes_flatten.append(param_flatten.size()[0])

    param = []

    vector_split = torch.split(vector, param_sizes_flatten)
    for p, v in zip(network.linear.parameters(), vector_split):
        param.append(v.reshape(p.size()))

    return param



def orthogonal_projection(vector, basis_set): 
    # orthogonally project vector onto basis vector set 
    # (returned vector is orthogonal to all the basis vectors)
    # basis_set must be 2d tensor
    # all basis must be unit vector (to skip dominator term(u_dot_u) in projection operator(v_dot_u / u_dot_u * u))
    # vector must be 1d tensor
    
    vector = vector.detach().clone().reshape(1, -1)

    projections = torch.matmul(basis_set, vector.T) # (i-1, p)*(p, 1)=(i-1, 1) # v_dot_U
    project_vectors = projections * basis_set # broadcasting
    project_vector = torch.sum(project_vectors, dim=0)
    vector -= project_vector
    vector.squeeze_()
    v_norm = torch.norm(vector, p=2.0, dim=0)
    if v_norm < 1e-3:
        vector = torch.zeros_like(vector)

    return vector



def check_orthogonality(basis_set, filename=None):

    eps = 1e-3

    if filename is not None:
        with open(filename, 'w') as f:
            for i, b1 in enumerate(basis_set):
                for j, b2 in enumerate(basis_set):
                    b1_dot_b2 = torch.dot(b1, b2)
                    f.write('({},{}): {}\n'.format(i, j, b1_dot_b2))
                    if i == j:
                        assert(b1_dot_b2 > 1 - eps and b1_dot_b2 < 1 + eps)
                    else:
                        assert(b1_dot_b2 < eps)
    else:
        for i, b1 in enumerate(basis_set):
            for j, b2 in enumerate(basis_set):
                b1_dot_b2 = torch.dot(b1, b2)
                print('({},{}): {}'.format(i, j, b1_dot_b2))
                if i == j:
                    assert(b1_dot_b2 > 1 - eps and b1_dot_b2 < 1 + eps)
                else:
                    assert(b1_dot_b2 < eps)

    pass



def compute_accuracy_matrix(config, task_id, test_subdatasets, accuracy_matrix, network, device):
    network.eval()

    for j in range(config.n_task):
        test_dataset = test_subdatasets[j]
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

        with torch.no_grad():
            for data, target in test_loader:
                prediction = network(data, j)
                pred_label = torch.argmax(prediction, dim=-1)
                correct = pred_label.eq(target).sum().item()
                accuracy_matrix[task_id, j] = 100. * correct / len(target)
                break



def compute_prediction_gradients(config, data, label, network, task_id):
    network.eval()

    print('compute the gradients of the prediction for the present task')

    gradient_vectors = torch.zeros(len(data), config.n_param, device=config.device)

    data = data.detach()
    for n, d in enumerate(tqdm(data)):
        d = torch.unflatten(d, 0, (1, -1))
        prediction = network(d, task_id)
        pred_gtl = prediction[0,label[n]]
        pred_gtl.backward()
        gradient_vectors[n] = network.body_grad_vector()
    
    return gradient_vectors.detach()