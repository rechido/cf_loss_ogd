import torch.nn as nn
from torchsummary import summary


class MLP(nn.Module): 

    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=100, n_hidden_layer=2):
        
        super(MLP, self).__init__()

        self.in_dim = in_channel*img_sz*img_sz
        layers = []
        layers += [nn.Linear(self.in_dim, hidden_dim)]
        #layers += [nn.BatchNorm1d(hidden_dim)]
        layers += [nn.ReLU(inplace=True)]
        for i in range(n_hidden_layer-1):
            layers += [nn.Linear(hidden_dim, hidden_dim)]
            #layers += [nn.BatchNorm1d(hidden_dim)]
            layers += [nn.ReLU(inplace=True)]
        layers += [nn.Linear(hidden_dim, out_dim)]

        self.linear = nn.Sequential(*layers)

        pass

    def forward(self, x):
        x = self.linear(x.view(-1,self.in_dim))
        return x



class CNN(nn.Module): 

    def __init__(self, in_channel=1, out_dim=10, img_size=32, feature_dim=2, n_feature_layer=1, hidden_dim=100, n_hidden_layer=1):
        
        super(CNN, self).__init__()

        threshold_ReLU = 0.1
        in_dim = in_channel * img_size * img_size

        feature_layers = []
        feature_layers += [nn.Conv2d(in_channel, feature_dim, kernel_size=3, stride=2, padding=1, bias=True)]
        feature_layers += [nn.BatchNorm2d(feature_dim),]
        feature_layers += [nn.LeakyReLU(threshold_ReLU, inplace=True)]
        feature_dim_in = feature_dim
        feature_dim_out = feature_dim * 2
        img_size = img_size // 2
        in_dim = feature_dim * img_size * img_size
        for i in range(n_feature_layer-1):
            feature_layers += [nn.Conv2d(feature_dim_in, feature_dim_out, kernel_size=3, stride=2, padding=1, bias=True)]
            feature_layers += [nn.BatchNorm2d(feature_dim_out),]
            feature_layers += [nn.LeakyReLU(threshold_ReLU, inplace=True)]
            img_size = img_size // 2
            in_dim = feature_dim_out * img_size * img_size
            feature_dim_in *= 2
            feature_dim_out *= 2            
        
        self.feature = nn.Sequential(*feature_layers)

        layers = []
        layers += [nn.Linear(in_dim, hidden_dim)]
        #layers += [nn.BatchNorm1d(hidden_dim)]
        layers += [nn.ReLU(inplace=True)]
        for i in range(n_hidden_layer-1):
            layers += [nn.Linear(hidden_dim, hidden_dim)]
            #layers += [nn.BatchNorm1d(hidden_dim)]
            layers += [nn.ReLU(inplace=True)]
        layers += [nn.Linear(hidden_dim, out_dim)]

        self.classifier = nn.Sequential(*layers)

        pass

    def forward(self, x):

        z = self.feature(x)
        z_ = nn.Flatten()(z)
        y = self.classifier(z_)

        return y


def create_model(config):
    
    if config.model == 'MLP':
        network = MLP(hidden_dim=config.hidden_dim, n_hidden_layer=config.n_hidden_layer).to(config.device)
    elif config.model == 'CNN':
        network = CNN(feature_dim=config.feature_dim, n_feature_layer=config.n_feature_layer, hidden_dim=config.hidden_dim, n_hidden_layer=config.n_hidden_layer).to(config.device)
    summary(network, input_size=(1,32,32))

    return network

