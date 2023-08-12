import torch
import torch.nn as nn
from torch.nn.utils.convert_parameters import  parameters_to_vector, vector_to_parameters



class Model(nn.Module):

    def __init__(self, model_type, out_dim=10, in_channel=1, img_sz=32, hidden_dim=100, n_hidden_layer=2, n_head=1, conv1_channel=20, conv2_channel=50):
        
        super(Model, self).__init__()

        self.out_dim = out_dim
        self.in_channel = in_channel
        self.img_sz = img_sz
        self.hidden_dim = hidden_dim
        self.n_hidden_layer = n_hidden_layer
        self.n_head = n_head
        self.conv1_channel = conv1_channel
        self.conv2_channel = conv2_channel

        if model_type == 'MLP':
            self.MLP()
        elif model_type == 'Lenet':
            self.Lenet()
        else:
            assert False, 'Wrong model_type'

        pass

    def MLP(self):

        self.conv = None

        self.in_dim = self.in_channel * self.img_sz * self.img_sz
        layers = []
        layers += [nn.Linear(self.in_dim, self.hidden_dim)]
        layers += [nn.ReLU(inplace=True)]
        for i in range(self.n_hidden_layer - 1):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim)]
            layers += [nn.ReLU(inplace=True)]

        self.linear = nn.Sequential(*layers)

        self.heads = nn.ModuleDict()
        for head_id in range(self.n_head):
            self.heads[str(head_id)] = nn.Linear(self.hidden_dim, self.out_dim) 

        pass

    def Lenet(self):

        feat_map_sz = self.img_sz//4
        self.in_dim = self.conv2_channel * feat_map_sz * feat_map_sz

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.conv1_channel, 5, padding=2),
            nn.BatchNorm2d(self.conv1_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.conv1_channel, self.conv2_channel, 5, padding=2),
            nn.BatchNorm2d(self.conv2_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )

        self.linear = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.heads = nn.ModuleDict()
        for i in range(self.n_head):
            self.heads[str(i)] = nn.Linear(self.hidden_dim, self.out_dim) 

        pass

    def forward(self, x, task_ids=None):
        if self.n_head == 1 or task_ids == None:
            task_ids = torch.zeros(x.shape[0])
        if self.conv is not None:
            x = self.conv(x)
        x = self.linear(x.view(-1,self.in_dim))
        y = torch.zeros(x.shape[0], self.out_dim, device=x.device)
        for head_id in range(self.n_head):
            y_ = self.heads[str(head_id)](x)
            y[task_ids == head_id] = y_[task_ids == head_id]
        return y
    
    def body_param_vector(self, skip_conv=True):

        if skip_conv:
            return parameters_to_vector(self.linear.parameters())

        else:
            return parameters_to_vector(list(self.conv.parameters()) + list(self.linear.parameters()))

    def head_param_vector(self, head_id=0):
        if self.n_head == 1:
            head_id = 0

        return parameters_to_vector(self.heads[str(head_id)].parameters())
    
    def body_grad_vector(self, skip_conv=True):
        vector = []
        device = None

        if skip_conv == False:
            for p in self.conv.parameters():
                vector.append(p.grad.view(-1))

        for p in self.linear.parameters():
            if device is None:
                device = p.device
            vector.append(p.grad.view(-1))

        return torch.cat(vector).to(device)
    
    def head_grad_vector(self, head_id=0):
        if self.n_head == 1:
            head_id = 0

        vector = []
        device = None

        for p in self.heads[str(head_id)].parameters():
            if device is None:
                device = p.device
            vector.append(p.grad.view(-1))

        return torch.cat(vector).to(device)
    
    def update_body(self, param_vector, skip_conv=True):

        if skip_conv:
            vector_to_parameters(param_vector, self.linear.parameters())
            
        else:
            vector_to_parameters(param_vector, list(self.conv.parameters()) + list(self.linear.parameters()))
            
    def update_head(self, param_vector, head_id=0):
        if self.n_head == 1:
            head_id = 0

        vector_to_parameters(param_vector, self.heads[str(head_id)].parameters())

    def get_parameters(self):
        if self.conv is not None:
            return list(self.conv.parameters()) + list(self.linear.parameters())
        else:
            return list(self.linear.parameters())


    



