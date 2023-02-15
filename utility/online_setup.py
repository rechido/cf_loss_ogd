import torch
from torch.utils.data import DataLoader, TensorDataset
from utility.visualize import plot_image_grid 

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
        for i in range(config.n_class):
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

    print('number of tasks = {}'.format(len(train_subdatasets)))

    print()
    print("Train Data")

    for i, subdataset in enumerate(train_subdatasets):

        print("task={}, Data Size={}".format(i, len(subdataset)))

        data_loader = DataLoader(subdataset, batch_size=128, shuffle=True, drop_last=True)

        for x, y in data_loader:
            plot_image_grid(x, 2, 10)
            break

    print()
    print("Test Data")

    for i, subdataset in enumerate(test_subdatasets):

        print("task={}, Data Size={}".format(i, len(subdataset)))

        data_loader = DataLoader(subdataset, batch_size=128, shuffle=True, drop_last=True)

        for x, y in data_loader:
            plot_image_grid(x, 2, 10)
            break

    return train_subdatasets, test_subdatasets

