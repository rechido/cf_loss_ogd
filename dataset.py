import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST, CIFAR100
from torchvision import transforms



def mnist(config):

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
    
    return train_dataset, test_dataset



def cifar_100(config):

     train_dataset = CIFAR100(config.dataset_path, train=True, download=False,
                              transform=transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
         ]))
     
     test_dataset = CIFAR100(config.dataset_path, train=False, download=False,
                             transform=transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
         ]))
     
     return train_dataset, test_dataset



def merge(subdatasets):
    datas = []
    targets = []

    for subdataset in subdatasets:
        x, y = subdataset[:]
        datas.append(x)
        targets.append(y)

    data = torch.cat(datas)
    target = torch.cat(targets)

    return TensorDataset(data, target)



def make_permuted_mnist(config, train_dataset, test_dataset):
    # create data stream for online learning

    train_subdatasets = []

    data_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

    for x, y in data_loader:
        for i in range(10):
            data = x[y == i][:config.subset_size_per_class]
            labels = y[y == i][:config.subset_size_per_class]
            train_subdatasets.append(TensorDataset(data, labels))
        break

    train_dataset = merge(train_subdatasets)

    train_subdatasets = []
    test_subdatasets = []

    # task 1 is original mnist
    train_subdatasets.append(train_dataset)

    data_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    for x, y in data_loader:
        test_subdatasets.append(TensorDataset(x, y))
        break

    # make permuted datasets for the later tasks
    for _ in range(config.n_task-1):

        data_loader = DataLoader(train_dataset, batch_size=config.subset_size, shuffle=True, drop_last=False)
        for x, y in data_loader:
            data = x
            targets = y
            break

        test_data_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, drop_last=False)
        for x, y in test_data_loader:
            test_data = x
            test_targets = y
            break

        size = data.shape[-2:]
        index=torch.randperm(size.numel()) # same permute rule should be applied to train & test data
        
        data_flatten = torch.flatten(data, start_dim=-2, end_dim=-1)
        data_flatten_permuted = torch.index_select(data_flatten, dim=-1, index=index)
        data_permuted = torch.nn.Unflatten(-1, size)(data_flatten_permuted)
        train_subdatasets.append(TensorDataset(data_permuted, targets))

        
        test_data_flatten = torch.flatten(test_data, start_dim=-2, end_dim=-1)
        test_data_flatten_permuted = torch.index_select(test_data_flatten, dim=-1, index=index)
        test_data_permuted = torch.nn.Unflatten(-1, size)(test_data_flatten_permuted)
        test_subdatasets.append(TensorDataset(test_data_permuted, test_targets))

    return train_subdatasets, test_subdatasets



def make_split_mnist(config, train_dataset, test_dataset):
    # create data stream for online learning

    train_subdatasets = []
    test_subdatasets = []

    data_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

    for x, y in data_loader:
        for i in range(5):
            class_even = 2 * i
            class_odd = 2 * i + 1

            data_even = x[y == class_even][:config.subset_size_per_class]
            labels_even = y[y == class_even][:config.subset_size_per_class] % 2
            dataset_even = TensorDataset(data_even, labels_even)

            data_odd = x[y == class_odd][:config.subset_size_per_class]
            labels_odd = y[y == class_odd][:config.subset_size_per_class] % 2
            dataset_odd = TensorDataset(data_odd, labels_odd)

            train_subdatasets.append(merge([dataset_even, dataset_odd]))
        break

    data_loader = DataLoader(test_dataset, batch_size=len(train_dataset), shuffle=True)

    for x, y in data_loader:
        for i in range(5):
            class_even = 2 * i
            class_odd = 2 * i + 1

            data_even = x[y == class_even]
            labels_even = y[y == class_even] % 2
            dataset_even = TensorDataset(data_even, labels_even)

            data_odd = x[y == class_odd]
            labels_odd = y[y == class_odd] % 2
            dataset_odd = TensorDataset(data_odd, labels_odd)

            test_subdatasets.append(merge([dataset_even, dataset_odd]))
        break

    return train_subdatasets, test_subdatasets



def make_split_cifar(config, train_dataset, test_dataset):
    # create data stream for online learning

    train_subdatasets = []
    test_subdatasets = []

    data_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

    for x, y in data_loader:
        for i in range(20):
            class_0 = 5 * i
            class_1 = 5 * i + 1
            class_2 = 5 * i + 2
            class_3 = 5 * i + 3
            class_4 = 5 * i + 4

            data_0 = x[y == class_0][:config.subset_size_per_class]
            data_1 = x[y == class_1][:config.subset_size_per_class]
            data_2 = x[y == class_2][:config.subset_size_per_class]
            data_3 = x[y == class_3][:config.subset_size_per_class]
            data_4 = x[y == class_4][:config.subset_size_per_class]

            labels_0 = y[y == class_0][:config.subset_size_per_class] % 5
            labels_1 = y[y == class_1][:config.subset_size_per_class] % 5
            labels_2 = y[y == class_2][:config.subset_size_per_class] % 5
            labels_3 = y[y == class_3][:config.subset_size_per_class] % 5
            labels_4 = y[y == class_4][:config.subset_size_per_class] % 5

            dataset_0 = TensorDataset(data_0, labels_0)
            dataset_1 = TensorDataset(data_1, labels_1)
            dataset_2 = TensorDataset(data_2, labels_2)
            dataset_3 = TensorDataset(data_3, labels_3)
            dataset_4 = TensorDataset(data_4, labels_4)

            train_subdatasets.append(merge([dataset_0, dataset_1, dataset_2, dataset_3, dataset_4]))
        break

    data_loader = DataLoader(test_dataset, batch_size=len(train_dataset), shuffle=True)

    for x, y in data_loader:
        for i in range(20):
            class_0 = 5 * i
            class_1 = 5 * i + 1
            class_2 = 5 * i + 2
            class_3 = 5 * i + 3
            class_4 = 5 * i + 4

            data_0 = x[y == class_0]
            data_1 = x[y == class_1]
            data_2 = x[y == class_2]
            data_3 = x[y == class_3]
            data_4 = x[y == class_4]

            labels_0 = y[y == class_0] % 5
            labels_1 = y[y == class_1] % 5
            labels_2 = y[y == class_2] % 5
            labels_3 = y[y == class_3] % 5
            labels_4 = y[y == class_4] % 5

            dataset_0 = TensorDataset(data_0, labels_0)
            dataset_1 = TensorDataset(data_1, labels_1)
            dataset_2 = TensorDataset(data_2, labels_2)
            dataset_3 = TensorDataset(data_3, labels_3)
            dataset_4 = TensorDataset(data_4, labels_4)

            test_subdatasets.append(merge([dataset_0, dataset_1, dataset_2, dataset_3, dataset_4]))
        break

    return train_subdatasets, test_subdatasets



def make_dataset(config):

    if config.dataset == 'permuted_mnist':
        config.subset_size_per_class = 1000
        config.subset_size = config.subset_size_per_class * 10
        config.out_dim = 10
        config.n_head = 1
        config.in_channel = 1
        train_dataset, test_dataset = mnist(config)
        train_subdatasets, test_subdatasets = make_permuted_mnist(config, train_dataset, test_dataset)

    elif config.dataset == 'split_mnist':
        config.subset_size_per_class = 1000
        config.subset_size = config.subset_size_per_class * 2
        config.out_dim = 2
        config.n_head = config.n_task
        config.in_channel = 1
        train_dataset, test_dataset = mnist(config)
        train_subdatasets, test_subdatasets = make_split_mnist(config, train_dataset, test_dataset)

    elif config.dataset == 'split_cifar':
        config.subset_size_per_class = 500
        config.subset_size = config.subset_size_per_class * 5
        config.out_dim = 5
        config.n_head = config.n_task
        config.in_channel = 3
        train_dataset, test_dataset = cifar_100(config)
        train_subdatasets, test_subdatasets = make_split_cifar(config, train_dataset, test_dataset)

    else:
        assert False, 'Invalid Dataset'

    config.img_size = 32

    return train_subdatasets, test_subdatasets