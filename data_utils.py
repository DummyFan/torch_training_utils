import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, random_split


def _to_numpy(values):
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _labels_to_indices(labels):
    labels = _to_numpy(labels)
    if labels.ndim > 1:
        if labels.shape[1] == 1:
            labels = labels.reshape(-1)
        else:
            labels = np.argmax(labels, axis=1)
    return labels.astype(np.int64)


def _split_lengths(n_samples, split_ratio, expected_parts):
    if len(split_ratio) != expected_parts:
        raise ValueError(f"split_ratio must contain {expected_parts} values")

    ratio = np.asarray(split_ratio, dtype=float)
    if np.any(ratio < 0):
        raise ValueError("split_ratio cannot contain negative values")

    if np.isclose(ratio.sum(), n_samples) and np.all(np.equal(ratio, np.floor(ratio))):
        lengths = ratio.astype(int).tolist()
    elif np.isclose(ratio.sum(), 1.0):
        raw_lengths = ratio * n_samples
        lengths = np.floor(raw_lengths).astype(int)
        remainder = n_samples - int(lengths.sum())
        if remainder > 0:
            order = np.argsort(raw_lengths - lengths)[::-1]
            for idx in order[:remainder]:
                lengths[idx] += 1
        lengths = lengths.tolist()
    else:
        raise ValueError(
            "split_ratio must be integer lengths summing to n_samples or ratios summing to 1"
        )

    if sum(lengths) != n_samples:
        raise ValueError("split lengths do not sum to the dataset size")
    if any(length == 0 for length in lengths):
        raise ValueError("all split lengths must be greater than zero")
    return lengths


def _random_split(dataset, lengths, seed):
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, lengths, generator=generator)


def _stratified_subsets(dataset, labels, lengths, seed):
    indices = np.arange(len(dataset))
    labels = _labels_to_indices(labels)

    if len(lengths) == 2:
        train_idx, test_idx = train_test_split(
            indices,
            train_size=lengths[0],
            test_size=lengths[1],
            stratify=labels,
            random_state=seed,
        )
        return Subset(dataset, train_idx.tolist()), Subset(dataset, test_idx.tolist())

    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices,
        labels,
        train_size=lengths[0],
        test_size=lengths[1] + lengths[2],
        stratify=labels,
        random_state=seed,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=lengths[1],
        test_size=lengths[2],
        stratify=temp_labels,
        random_state=seed,
    )
    return (
        Subset(dataset, train_idx.tolist()),
        Subset(dataset, val_idx.tolist()),
        Subset(dataset, test_idx.tolist()),
    )


def _split_dataset(dataset, labels, split_ratio, expected_parts, seed, stratify):
    lengths = _split_lengths(len(dataset), split_ratio, expected_parts)
    if stratify:
        try:
            return _stratified_subsets(dataset, labels, lengths, seed)
        except ValueError as exc:
            warnings.warn(
                f"Could not create a stratified split ({exc}); falling back to random_split.",
                RuntimeWarning,
                stacklevel=2,
            )
    return _random_split(dataset, lengths, seed)


class MyDataset(Dataset):
    def __init__(self, data, labels, extra_dim=None, one_hot=False, num_classes=None):
        self.data = torch.as_tensor(data, dtype=torch.float32)
        if extra_dim is not None:
            self.data = torch.unsqueeze(self.data, extra_dim)

        labels = torch.as_tensor(labels)
        if one_hot:
            if labels.ndim > 1 and labels.shape[1] > 1:
                self.labels = labels.float()
            else:
                self.labels = nn.functional.one_hot(
                    labels.long().reshape(-1), num_classes=num_classes
                ).float()
        else:
            if labels.ndim > 1 and labels.shape[1] > 1:
                labels = labels.argmax(dim=1)
            self.labels = labels.long().reshape(-1)

        if len(self.data) != len(self.labels):
            raise ValueError(
                "data and labels must have the same first dimension after extra_dim is applied"
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_1d_labels(self):
        return _labels_to_indices(self.labels)

    def get_np_data(self):
        return _to_numpy(self.data)


def get_class_distribution(full_labels, subset, name):
    print(f"{name} set distribution")
    labels = _labels_to_indices(full_labels)
    subset_labels = labels[subset.indices]
    values, counts = np.unique(subset_labels, return_counts=True)
    for value, count in zip(values, counts):
        print(f"{value}: {count}")


def get_loaders(
    data,
    labels,
    batch_size,
    verbose=False,
    split_ratio=(0.7, 0.15, 0.15),
    seed=42,
    stratify=True,
    one_hot=False,
    num_classes=None,
    extra_dim=None,
):
    dataset = MyDataset(
        data, labels, extra_dim=extra_dim, one_hot=one_hot, num_classes=num_classes
    )
    train, val, test = _split_dataset(dataset, labels, split_ratio, 3, seed, stratify)

    if verbose:
        for subset, name in zip([train, val, test], ["Train", "Validation", "Test"]):
            get_class_distribution(labels, subset, name)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def get_train_test_loaders(
    data,
    labels,
    batch_size,
    verbose=False,
    split_ratio=(0.7, 0.3),
    seed=42,
    stratify=True,
    one_hot=False,
    num_classes=None,
    extra_dim=None,
):
    dataset = MyDataset(
        data, labels, extra_dim=extra_dim, one_hot=one_hot, num_classes=num_classes
    )
    train, test = _split_dataset(dataset, labels, split_ratio, 2, seed, stratify)

    if verbose:
        for subset, name in zip([train, test], ["Train", "Test"]):
            get_class_distribution(labels, subset, name)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
