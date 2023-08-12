import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F

class Episodic_Memory_Buffer(Dataset):
    def __init__(self):
        self.data = torch.empty(0)
        self.labels = torch.empty(0)
        self.task_ids = torch.empty(0)
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.task_ids[idx]

    def add(self, data, labels, task_id):
        task_ids = torch.full_like(labels, task_id)
        self.data = torch.cat((self.data, data))
        self.labels = torch.cat((self.labels, labels))
        self.task_ids = torch.cat((self.task_ids, task_ids))
        pass



class Experience_Replay(object):

    def __init__(self):
        self.buffer = Episodic_Memory_Buffer()
        pass

    def consolidate(self, dataset, config, task_id):
        print('Consolidate Examplar into replay memory...')
        data_loader = DataLoader(dataset, batch_size=config.n_examplar, shuffle=True)
        data, labels = next(iter(data_loader))
        self.buffer.add(data, labels, task_id)
        pass

    def penalty_loss(self, network, config):
        data_loader = DataLoader(self.buffer, batch_size=len(self.buffer), shuffle=True)
        data, labels, task_ids = next(iter(data_loader))
        data = data.to(config.device)
        labels = labels.long().to(config.device)
        task_ids = task_ids.to(config.device)
        outputs = network(data, task_ids)
        loss = F.cross_entropy(outputs, labels)
        return loss



class AGEM(object):

    def __init__(self):
        self.buffer = Episodic_Memory_Buffer()
        pass

    def consolidate(self, dataset, config, task_id):
        print('Consolidate Examplar into AGEM memory...')
        data_loader = DataLoader(dataset, batch_size=config.n_examplar, shuffle=True)
        data, labels = next(iter(data_loader))
        self.buffer.add(data, labels, task_id)
        pass

    def compute_gem_grad_vector(self, network, config, skip_conv=True):
        data_loader = DataLoader(self.buffer, batch_size=len(self.buffer), shuffle=True)
        data, labels, task_ids = next(iter(data_loader))
        data = data.to(config.device)
        labels = labels.long().to(config.device)
        task_ids = task_ids.to(config.device)
        outputs = network(data, task_ids)
        loss = F.cross_entropy(outputs, labels)
        network.zero_grad()
        loss.backward()
        return network.body_grad_vector(skip_conv)
    
    def orthogonal_projection(self, batch_grad_vector, network, config, skip_conv=True):
        gem_grad_vector = self.compute_gem_grad_vector(network, config, skip_conv)
        if torch.dot(batch_grad_vector, gem_grad_vector) >= 0:
            return batch_grad_vector
        new_grad_vector = batch_grad_vector - torch.dot(batch_grad_vector, gem_grad_vector) / torch.dot(gem_grad_vector, gem_grad_vector) * gem_grad_vector # orthogonal projection
        return new_grad_vector
    
