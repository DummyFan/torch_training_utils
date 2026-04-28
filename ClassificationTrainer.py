import os
from csv import writer
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch_training_utils.utils import set_device
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from datetime import datetime


class ClassificationTrainer:
    """
    A flexible training loop manager for classification models using PyTorch.

    Parameters
    ----------
    model : torch.nn.Module
        The classification model to be trained.

    criterion : callable
        Loss function used for optimization.

    optimizer : torch.optim.Optimizer
        Torch optimizer used to update model parameters.

    metric : callable
        Evaluation metric (e.g., accuracy_score, f1_score).

    metric_params : dict
        Additional parameters passed to the metric function.

    save_paths : dict
        Dictionary containing paths to save logs, models, and figures. Keys should include:
        'log_path', 'model_path', and 'fig_path'.
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        metric,
        device_id=None,
        metric_params=None,
        save_paths=None,
        task="multiclass",
        threshold=0.5,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metric = metric
        self.metric_params = {} if metric_params is None else metric_params
        self.task = task
        self.threshold = threshold
        self._valid_tasks = {"multiclass", "binary", "multilabel"}
        if self.task not in self._valid_tasks:
            raise ValueError(
                f"task must be one of {sorted(self._valid_tasks)}, got {task!r}"
            )

        self.device = set_device(device_id)
        self.model.to(self.device)
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if save_paths is None:
            save_paths = {
                "log_path": Path.cwd() / "logs",
                "model_path": Path.cwd() / "models",
                "fig_path": Path.cwd() / "figures",
            }
        self.log_path = Path(save_paths["log_path"])
        self.model_path = Path(save_paths["model_path"])
        self.fig_path = Path(save_paths["fig_path"])
        for path in (self.log_path, self.model_path, self.fig_path):
            path.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = self.model_path / f"{self.start_time}.pt"
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def _unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 0:
                raise ValueError("Received an empty batch from the dataloader")
            if len(batch) == 1:
                return batch[0], None
            return batch[0], batch[1]
        return batch, None

    def _require_targets(self, targets, method_name):
        if targets is None:
            raise ValueError(
                f"{method_name} requires dataloader batches to include targets"
            )

    def _batch_size(self, data):
        return data.size(0) if hasattr(data, "size") else len(data)

    def _prepare_targets_for_loss(self, outputs, targets):
        if self.task == "multiclass":
            if targets.shape == outputs.shape:
                return targets.float()
            targets = targets.long()
            if (
                targets.ndim == outputs.ndim
                and targets.ndim > 1
                and targets.size(1) == 1
            ):
                targets = targets.squeeze(1)
            return targets

        targets = targets.float()
        if (
            self.task == "binary"
            and targets.shape != outputs.shape
            and targets.numel() == outputs.numel()
        ):
            targets = targets.reshape_as(outputs)
        return targets

    def _labels_from_targets(self, targets, outputs=None):
        if self.task == "multiclass":
            if outputs is not None and targets.shape == outputs.shape:
                labels = targets.argmax(dim=1)
            else:
                labels = targets
                if labels.ndim > 1 and labels.size(1) == 1:
                    labels = labels.squeeze(1)
            return labels.detach().cpu().long().reshape(-1).tolist()

        if self.task == "binary":
            labels = (targets.float().squeeze() >= self.threshold).long()
            return labels.detach().cpu().reshape(-1).tolist()

        labels = (targets.float() >= self.threshold).long()
        return labels.detach().cpu().tolist()

    def _outputs_to_probabilities(self, outputs):
        if self.task == "multiclass":
            if outputs.ndim < 2 or outputs.size(1) == 1:
                return torch.sigmoid(outputs)
            return torch.softmax(outputs, dim=1)
        return torch.sigmoid(outputs)

    def _labels_from_outputs(self, outputs):
        if self.task == "multiclass":
            if outputs.ndim < 2 or outputs.size(1) == 1:
                labels = (torch.sigmoid(outputs).squeeze() >= self.threshold).long()
            else:
                labels = outputs.argmax(dim=1)
            return labels.detach().cpu().reshape(-1).tolist()

        probabilities = self._outputs_to_probabilities(outputs)
        labels = (probabilities >= self.threshold).long()
        if self.task == "binary":
            return labels.detach().cpu().reshape(-1).tolist()
        return labels.detach().cpu().tolist()

    def _loss_contribution(self, loss, batch_size):
        reduction = getattr(self.criterion, "reduction", "mean")
        if reduction == "sum":
            return loss.item()
        return loss.item() * batch_size

    def _average_loss(self, loss_total, sample_total):
        if sample_total == 0:
            return np.inf
        loss = loss_total / sample_total
        return np.inf if np.isnan(loss) else loss

    def load_data(self, train_dataset, test_dataset, val_dataset=None, batch_size=64):
        """
        Loads and prepares data for training, validation, and testing.

        Parameters
        ----------
        train_dataset : torch.utils.data.Dataset
            Torch Dataset used for training.

        test_dataset : torch.utils.data.Dataset
            Torch Dataset used for testing.

        val_dataset : torch.utils.data.Dataset, optional
            Torch Dataset used for validation.

        batch_size : int, default=64
            Batch size for all dataloaders.
        """
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

    def train(self, data_loader):
        """
        Performs a single epoch of training.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            Torch Dataloader for training data.

        Returns
        -------
        train_loss : float
            Average training loss.

        train_score : float
            Evaluation metric score on the training data.
        """
        self.model.train()
        loss_total = 0.0
        sample_total = 0
        y_true, y_pred = [], []

        for batch in data_loader:
            data, targets = self._unpack_batch(batch)
            self._require_targets(targets, "train")
            data, targets = data.to(self.device), targets.to(self.device)
            outputs = self.model(data)

            loss = self.criterion(
                outputs, self._prepare_targets_for_loss(outputs, targets)
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = self._batch_size(data)
            y_true.extend(self._labels_from_targets(targets, outputs))
            y_pred.extend(self._labels_from_outputs(outputs))
            loss_total += self._loss_contribution(loss, batch_size)
            sample_total += batch_size

        train_loss = self._average_loss(loss_total, sample_total)
        train_score = self.metric(y_true, y_pred, **self.metric_params)
        return train_loss, train_score

    def validate(self, data_loader):
        """
        Evaluates the model on a validation set.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            Torch Dataloader for validation data.

        Returns
        -------
        val_loss : float
            Average validation loss.

        val_score : float
            Evaluation metric score on the validation data.
        """
        self.model.eval()
        loss_total = 0.0
        sample_total = 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in data_loader:
                data, targets = self._unpack_batch(batch)
                self._require_targets(targets, "validate")
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)

                loss = self.criterion(
                    outputs, self._prepare_targets_for_loss(outputs, targets)
                )
                batch_size = self._batch_size(data)
                loss_total += self._loss_contribution(loss, batch_size)
                sample_total += batch_size

                y_true.extend(self._labels_from_targets(targets, outputs))
                y_pred.extend(self._labels_from_outputs(outputs))

        val_loss = self._average_loss(loss_total, sample_total)
        val_score = self.metric(y_true, y_pred, **self.metric_params)
        return val_loss, val_score

    def test(self, data_loader, plot_confusion=False):
        """
        Tests the model on the test set.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            Torch Dataloader for test data.

        Returns
        -------
        test_loss : float
            Average test loss.

        test_score : float
            Evaluation metric score on the test data.

        y_out : list
            Model output probabilities, provided for purposes like computing the AUC score.
        """
        self.model.eval()
        y_true, y_pred, y_out = [], [], []
        loss_total = 0.0
        sample_total = 0

        with torch.no_grad():
            for batch in data_loader:
                data, targets = self._unpack_batch(batch)
                self._require_targets(targets, "test")
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)

                loss = self.criterion(
                    outputs, self._prepare_targets_for_loss(outputs, targets)
                )
                batch_size = self._batch_size(data)
                loss_total += self._loss_contribution(loss, batch_size)
                sample_total += batch_size

                y_true.extend(self._labels_from_targets(targets, outputs))
                y_pred.extend(self._labels_from_outputs(outputs))
                y_out.extend(self._outputs_to_probabilities(outputs).cpu().tolist())

        test_loss = self._average_loss(loss_total, sample_total)
        test_score = self.metric(y_true, y_pred, **self.metric_params)
        if plot_confusion:
            self.plot_confusion_matrix(y_true, y_pred)
        return test_loss, test_score, y_pred, y_out

    def predict(self, data_loader, return_prob=False):
        """
        Generates predictions from the model.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            Torch Dataloader for inference data.

        return_prob : bool, default=False
            If True, returns class probabilities instead of predicted classes.

        Returns
        -------
        y_pred : list
            Predicted class indices or probabilities.
        """
        self.model.eval()
        y_pred = []

        with torch.no_grad():
            for batch in data_loader:
                data, _ = self._unpack_batch(batch)
                data = data.to(self.device)
                outputs = self.model(data)
                if not return_prob:
                    y_pred.extend(self._labels_from_outputs(outputs))
                else:
                    y_pred.extend(
                        self._outputs_to_probabilities(outputs).cpu().tolist()
                    )

        return y_pred

    def fit_holdout(
        self,
        epochs,
        patience=10,
        print_results=True,
        divergence_patience=5,
        max_loss=100.0,
        warmup_epochs=5,
    ):
        """
        Trains the model using a holdout validation set with early stopping.

        Parameters
        ----------
        epochs : int
            Maximum number of training epochs.

        patience : int, default=10
            Number of epochs to wait for improvement in validation loss before early stopping.

        print_results : bool, default=True
            Whether to print early stopping and final test results.

        divergence_patience : int, default=5
            Number of consecutive divergent epochs allowed before stopping. A divergent epoch
            has a non-finite loss, or a loss greater than max_loss after warmup.

        max_loss : float or None, default=100.0
            Maximum finite train or validation loss allowed after warmup. If None, only
            non-finite losses are treated as divergent.

        warmup_epochs : int, default=0
            Number of initial epochs to run before applying normal patience-based early
            stopping or max_loss divergence checks. Non-finite losses are still checked.
        """
        if self.train_loader is None:
            raise ValueError("Call load_data before fit_holdout")
        if self.val_loader is None:
            raise ValueError("fit_holdout requires a validation dataset")
        if divergence_patience < 1:
            raise ValueError("divergence_patience must be at least 1")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs cannot be negative")
        if max_loss is not None and max_loss <= 0:
            raise ValueError("max_loss must be positive when provided")

        best_val_loss = np.inf
        best_val_score = 0
        counter = 0
        divergence_counter = 0
        self.train_losses, self.train_scores = [], []
        self.val_losses, self.val_scores = [], []
        self.stop_epoch = 0
        self.stop_reason = None
        # Save the initialized model first in case the criteria cannot be met
        torch.save(self.model.state_dict(), self.checkpoint_path)

        with tqdm(range(epochs), unit="epoch") as tepoch:
            for epoch in tepoch:
                train_loss, train_score = self.train(
                    self.train_loader
                )  # Model training
                val_loss, val_score = self.validate(self.val_loader)  # Model validation

                self.train_losses.append(train_loss)
                self.train_scores.append(train_score)
                self.val_losses.append(val_loss)
                self.val_scores.append(val_score)

                # Set progress bar info
                tepoch.set_postfix(
                    train_loss=train_loss,
                    train_score=train_score,
                    val_loss=val_loss,
                    val_score=val_score,
                )

                in_warmup = epoch < warmup_epochs
                non_finite_loss = not (
                    np.isfinite(train_loss) and np.isfinite(val_loss)
                )
                above_max_loss = (
                    max_loss is not None
                    and not in_warmup
                    and (train_loss > max_loss or val_loss > max_loss)
                )
                if non_finite_loss or above_max_loss:
                    divergence_counter += 1
                else:
                    divergence_counter = 0

                if divergence_counter >= divergence_patience:
                    self.stop_reason = "divergence"
                    if print_results:
                        print(f"Early stopped after {epoch + 1} epochs")
                        print("Reason: loss divergence")
                        print(f"Train loss: {train_loss}")
                        print(f"Val loss: {val_loss}")
                        if max_loss is not None:
                            print(f"Max loss threshold: {max_loss}")
                        print(f"Best result at epoch {self.stop_epoch + 1}")
                        print(f"Best val loss: {best_val_loss}")
                        print(f"Best val score: {best_val_score}\n")
                    break

                # Check early stopping criteria
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_score = val_score
                    counter = 0
                    self.stop_epoch = epoch
                    # Save model whenever validation loss reaches a new minimum
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                else:
                    if not in_warmup:
                        counter += 1

                # Terminates training at patience limit
                if not in_warmup and counter >= patience:
                    self.stop_reason = "patience"
                    if print_results:
                        print(f"Early stopped after {epoch + 1} epochs")
                        print("Reason: validation loss did not improve")
                        print(f"Best result at epoch {self.stop_epoch + 1}")
                        print(f"Best val loss: {best_val_loss}")
                        print(f"Best val score: {best_val_score}\n")
                    break

        self.best_val_loss = best_val_loss
        self.best_val_score = best_val_score

    def compute_test_results(self, print_results=True, plot_confusion=True):
        if self.test_loader is None:
            raise ValueError("Call load_data before compute_test_results")
        # Load the best model
        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device)
        )
        self.test_loss, self.test_score, self.test_pred, self.test_out = self.test(
            self.test_loader, plot_confusion=plot_confusion
        )

        if print_results:
            print(f"Test loss: {self.test_loss}")
            print(f"Test score: {self.test_score}")

    def record_results(self, log_file, log_dict):
        """
        Records the information from log_dict to log_file in log_path.

        Parameters
        ----------
        log_file : str
            Name of csv file to write to.

        log_dict : dict
            Dictionary containing information to be stored.
        """
        log_file_path = self.log_path / log_file
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(log_file_path):
            title = log_dict.keys()
            with open(log_file_path, "w", newline="") as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(title)

        log_record = log_dict.values()
        with open(log_file_path, "a", newline="") as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(log_record)

    def plot_learning_curves(self, metric_name, fig_size=(10, 5), mark_minimum=True):
        """
        Plots the learning curves for training and validation. optionally marks the epoch
        with the minimum validation loss for reference.

        Parameters
        ----------
        metric_name : str
            Name of the evaluation metric to be displayed on the plot (e.g., "Accuracy").

        fig_size : tuple of int, default=(10, 5)
            Size of the figure as (width, height).

        mark_minimum : bool, default=True
            Whether to mark the epoch with the minimum validation loss with a vertical line.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size, layout="constrained")
        fig.suptitle("Learning Curves")
        x = range(len(self.train_losses))
        ax1.plot(x, self.train_losses, label="Train")
        ax1.plot(x, self.val_losses, label="Validation")
        ax1.set_ylabel("Loss")
        ax1.set_xlabel("Epochs")
        ax2.plot(x, self.train_scores, label="Train")
        ax2.plot(x, self.val_scores, label="Validation")
        ax2.set_ylabel(metric_name)
        ax2.set_xlabel("Epochs")
        ax1.legend(loc="upper right")
        ax2.legend(loc="lower right")

        if mark_minimum:
            min_loss_pos = int(np.argmin(self.val_losses))
            ax1.axvline(x=min_loss_pos, color="red", linestyle="dashed")
            ax2.axvline(x=min_loss_pos, color="red", linestyle="dashed")

        plt.savefig(self.fig_path / f"{self.start_time}_learning_curve.png")
        plt.show()

    def plot_confusion_matrix(
        self,
        y_true,
        y_pred,
        categories="auto",
        xyticks=True,
        xyplotlabels=True,
        figsize=(4, 4),
        cmap="Blues",
        title=None,
    ):
        """
        Plot a confusion matrix using Seaborn heatmap visualization.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Ground truth (correct) target values.

        y_pred : array-like of shape (n_samples,)
            Estimated targets as returned by a classifier.

        categories : list of str or 'auto', default='auto'
            Labels to display on the axes. If 'auto', uses numeric class labels from the data.

        xyticks : bool, default=True
            Whether to display tick labels (categories) on the x and y axes.

        xyplotlabels : bool, default=True
            Whether to display axis labels ("True label" and "Predicted label").

        figsize : tuple of int, default=(5, 5)
            Size of the figure in inches (width, height).

        cmap : str or matplotlib Colormap, default='Blues'
            Colormap used for the heatmap. See `matplotlib.pyplot.colormaps()` for options.

        title : str or None, default=None
            Title for the confusion matrix plot.
        """
        cf = confusion_matrix(y_true, y_pred)
        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if not xyticks:
            # Do not show categories if xyticks is False
            categories = False

        # MAKE THE HEATMAP VISUALIZATION
        plt.figure(figsize=figsize)
        sns.heatmap(
            cf,
            fmt="",
            annot=True,
            cmap=cmap,
            cbar=False,
            xticklabels=categories,
            yticklabels=categories,
        )

        if xyplotlabels:
            plt.ylabel("True label")
            plt.xlabel("Predicted label")

        if title:
            plt.title(title)
        plt.savefig(self.fig_path / f"{self.start_time}_confusion_matrix.png")
        plt.show()
