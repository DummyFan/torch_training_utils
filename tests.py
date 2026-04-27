import tempfile
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

if __package__ is None or __package__ == '':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch_training_utils.ClassificationTrainer import ClassificationTrainer
from torch_training_utils.data_utils import MyDataset, get_train_test_loaders
from torch_training_utils.utils import seed_everything


def _save_paths(root):
    root = Path(root)
    return {
        'log_path': root / 'log',
        'model_path': root / 'model',
        'fig_path': root / 'figs',
    }


def test_mydataset_integer_labels_by_default():
    dataset = MyDataset(np.zeros((3, 2)), [0, 1, 1], extra_dim=1)
    data, label = dataset[0]

    assert data.shape == (1, 2)
    assert label.ndim == 0
    assert dataset.get_1d_labels().tolist() == [0, 1, 1]


def test_mydataset_accepts_extra_dim_zero_when_lengths_match():
    dataset = MyDataset(np.zeros(2), [0], extra_dim=0)

    assert dataset.data.shape == (1, 2)
    assert len(dataset) == 1


def test_mydataset_can_return_one_hot_labels():
    dataset = MyDataset(np.zeros((3, 2)), [0, 1, 1], one_hot=True, num_classes=2)
    _, label = dataset[0]

    assert label.shape == (2,)
    assert dataset.get_1d_labels().tolist() == [0, 1, 1]


def test_train_test_loaders_are_stratified():
    x = np.zeros((20, 2))
    y = np.array([0] * 10 + [1] * 10)
    train_loader, test_loader = get_train_test_loaders(
        x,
        y,
        batch_size=4,
        split_ratio=(0.5, 0.5),
        seed=42,
        stratify=True,
    )

    train_labels = y[train_loader.dataset.indices]
    test_labels = y[test_loader.dataset.indices]

    assert np.bincount(train_labels).tolist() == [5, 5]
    assert np.bincount(test_labels).tolist() == [5, 5]


def test_trainer_accepts_integer_cross_entropy_targets():
    seed_everything(42)
    x = torch.randn(12, 4)
    y = torch.tensor([0, 1, 2] * 4, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=5, shuffle=False)
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = ClassificationTrainer(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            metric=f1_score,
            metric_params={'average': 'macro', 'zero_division': 0.0},
            save_paths=_save_paths(tmp_dir),
        )
        loss, score = trainer.train(loader)
        probabilities = trainer.predict(loader, return_prob=True)

    assert np.isfinite(loss)
    assert 0.0 <= score <= 1.0
    assert len(probabilities) == len(y)
    assert np.allclose(np.sum(probabilities, axis=1), 1.0)


def test_fit_holdout_stops_after_divergence_patience_with_warmup():
    seed_everything(42)
    x = torch.randn(18, 4)
    y = torch.tensor([0, 1, 2] * 6, dtype=torch.long)
    train_dataset = TensorDataset(x[:12], y[:12])
    val_dataset = TensorDataset(x[12:15], y[12:15])
    test_dataset = TensorDataset(x[15:], y[15:])
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = ClassificationTrainer(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            metric=f1_score,
            metric_params={"average": "macro", "zero_division": 0.0},
            save_paths=_save_paths(tmp_dir),
        )
        trainer.load_data(
            train_dataset,
            test_dataset,
            val_dataset=val_dataset,
            batch_size=3,
        )
        trainer.fit_holdout(
            epochs=5,
            patience=5,
            print_results=False,
            divergence_patience=2,
            max_loss=0.01,
            warmup_epochs=1,
        )

    assert trainer.stop_reason == "divergence"
    assert len(trainer.train_losses) == 3


if __name__ == '__main__':
    test_mydataset_integer_labels_by_default()
    test_mydataset_accepts_extra_dim_zero_when_lengths_match()
    test_mydataset_can_return_one_hot_labels()
    test_train_test_loaders_are_stratified()
    test_trainer_accepts_integer_cross_entropy_targets()
    test_fit_holdout_stops_after_divergence_patience_with_warmup()
    print('All tests passed.')
