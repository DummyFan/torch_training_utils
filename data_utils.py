import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


# torch框架所需的数据集格式
class EEG_Dataset(Dataset):
    def __init__(self, data, labels, extra_dim=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        if extra_dim:
            self.data = torch.unsqueeze(self.data, extra_dim)
        self.labels = nn.functional.one_hot(torch.tensor(labels, dtype=torch.long))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    # 提取一维标签
    def get_1d_labels(self):
        labels = np.argmax(self.labels, axis=1)
        return labels

    # 提取numpy格式数据
    def get_np_data(self):
        return np.array(self.data)


# 展示各类别的数据分布情况
def get_class_distribution(full_labels, subset, name):
    print(f'{name} set distribution')
    labels = full_labels[subset.indices]
    values, counts = np.unique(labels, return_counts=True)
    for value, count in zip(values, counts):
        print(f'{value}: {count}')


# 分割训练、验证、测试集，装载torch dataloader
def get_loaders(data, labels, batch_size, verbose, split_ratio, seed=42):
    dataset = EEG_Dataset(data, labels)
    generator = torch.Generator().manual_seed(seed)
    # 分割训练验证测试集，比例通过split_ratio调整
    train, val, test = random_split(dataset, split_ratio, generator=generator)

    if verbose:
        for subset, name in zip([train, val, test], ['Train', 'Validation', 'Test']):
            get_class_distribution(labels, subset, name)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)
    return train_loader, val_loader, test_loader


# 分割训练、测试集，装载torch dataloader
def get_train_test_loaders(data, labels, batch_size, verbose, split_ratio, seed=42):
    dataset = EEG_Dataset(data, labels)
    generator = torch.Generator().manual_seed(seed)
    # 分割训练测试集，比例通过split_ratio调整
    train, test = random_split(dataset, split_ratio, generator=generator)

    if verbose:
        for subset, name in zip([train, test], ['Train', 'Test']):
            get_class_distribution(labels, subset, name)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size)
    return train_loader, test_loader
