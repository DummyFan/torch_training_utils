import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch_training_utils.utils import set_device
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


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

    def __init__(self, model, criterion, optimizer, metric, metric_params, save_paths):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metric = metric
        self.metric_params = metric_params
        self.device = set_device()
        self.model.to(self.device)

        self.log_path = save_paths['log_path']
        self.model_path = save_paths['model_path']
        self.fig_path = save_paths['fig_path']

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
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if val_dataset:
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
        train_loss = 0.0
        y_true, y_pred = [], []

        for data, targets in data_loader:
            data, targets = data.to(self.device), targets.to(self.device)
            outputs = self.model(data)

            loss = self.criterion(outputs, targets.float())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            true = targets.argmax(1)
            pred = outputs.argmax(1)

            y_true.extend(true.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            train_loss += loss.item()

        train_loss /= len(data_loader)
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
        val_loss = 0.0
        y_true, y_pred = [], []

        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)

                loss = self.criterion(outputs, targets.float())
                val_loss += loss.item()

                true = targets.argmax(1)
                pred = outputs.argmax(1)

                y_true.extend(true.cpu().tolist())
                y_pred.extend(pred.cpu().tolist())

        val_loss /= len(data_loader)
        val_score = self.metric(y_true, y_pred, **self.metric_params)
        return val_loss, val_score

    def test(self, data_loader):
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
            Raw model outputs (probabilities or logits), provided for purposes like computing the AUC score.
        """
        self.model.eval()
        y_true, y_pred, y_out = [], [], []
        test_loss = 0.0

        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)

                loss = self.criterion(outputs, targets.float())
                test_loss += loss.item()

                true = targets.argmax(1)
                pred = outputs.argmax(1)

                y_true.extend(true.cpu().tolist())
                y_pred.extend(pred.cpu().tolist())
                y_out.extend(outputs.cpu().tolist())

        test_score = self.metric(y_true, y_pred, **self.metric_params)
        self.plot_confusion_matrix(y_true, y_pred)
        return test_loss, test_score, y_out

    def predict(self, data_loader, return_prob=False):
        """
        Generates predictions from the model.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            Torch Dataloader for inference data.

        return_prob : bool, default=False
            If True, returns model output probabilities instead of predicted classes.

        Returns
        -------
        avg_loss : float
            Average loss across the dataset.

        y_pred : list
            Predicted class indices or raw probabilities.
        """
        self.model.eval()
        y_pred = []
        avg_loss = 0.0

        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)

                loss = self.criterion(outputs, targets.float())
                avg_loss += loss.item()

                if not return_prob:
                    pred = outputs.argmax(1)
                    y_pred.extend(pred.cpu().tolist())
                else:
                    y_pred.extend(outputs.cpu().tolist())

        avg_loss /= len(data_loader)
        return avg_loss, y_pred

    def fit_holdout(self, epochs, patience=10, print_results=True):
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
        """
        best_val_loss = np.inf
        best_val_score = 0
        counter = 0
        self.train_losses, self.train_scores = [], []
        self.val_losses, self.val_scores = [], []

        with tqdm(range(epochs), unit='epoch') as tepoch:
            for epoch in tepoch:
                train_loss, train_score = self.train(self.train_loader)  # Model training
                val_loss, val_score = self.validate(self.val_loader)  # Model validation

                self.train_losses.append(train_loss)
                self.train_scores.append(train_score)
                self.val_losses.append(val_loss)
                self.val_scores.append(val_score)

                # Set progress bar info
                tepoch.set_postfix(train_loss=train_loss, train_score=train_score,
                                   val_loss=val_loss, val_score=val_score)

                # Check early stopping criteria
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_score = val_score
                    counter = 0
                    # Save model whenever validation loss reaches a new minimum
                    torch.save(self.model.state_dict(), self.model_path / 'best_model.pt')
                else:
                    counter += 1

                # Terminates training at patience limit
                if counter >= patience:
                    if print_results:
                        print(f'Early stopped after {epoch + 1} epochs')
                        print(f'Best val loss: {best_val_loss}')
                        print(f'Best val score: {best_val_score}\n')
                    break

        # Load the best model
        self.model.load_state_dict(torch.load(self.model_path / 'best_model.pt'))
        self.test_loss, self.test_score, self.test_out = self.test(self.test_loader)

        if print_results:
            print(f'Test loss: {self.test_loss}')
            print(f'Test score: {self.test_score}')

    def plot_learning_curves(self, metric_name, fig_size=(10, 5), mark_minimum=True):
        """
        Plots the learning curves for training and validation.

        Parameters
        ----------
        metric_name : str
            Name of the evaluation metric to be displayed on the plot (e.g., "Accuracy").

        fig_size : tuple of int, default=(10, 5)
            Size of the figure as (width, height).

        mark_minimum : bool, default=True
            Whether to mark the epoch with the minimum validation loss with a vertical line.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
        fig.suptitle('Learning Curves')
        x = range(len(self.train_losses))
        ax1.plot(x, self.train_losses, label='Train')
        ax1.plot(x, self.val_losses, label='Validation')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epochs')
        ax2.plot(x, self.train_scores, label='Train')
        ax2.plot(x, self.val_scores, label='Validation')
        ax2.set_ylabel(metric_name)
        ax2.set_xlabel('Epochs')
        plt.legend(loc="lower right")

        if mark_minimum:
            min_loss_pos = np.argmin(self.val_losses)
            ax1.axvline(x=min_loss_pos, color='red', linestyle='dashed')
            ax2.axvline(x=min_loss_pos, color='red', linestyle='dashed')

        plt.savefig(self.fig_path / f'learning_curve.png')
        plt.show()

    def plot_confusion_matrix(
            self,
            y_true,
            y_pred,
            categories='auto',
            xyticks=True,
            xyplotlabels=True,
            figsize=(5, 5),
            cmap='Blues',
            title=None
    ):
        cf = confusion_matrix(y_true, y_pred)
        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if not xyticks:
            # Do not show categories if xyticks is False
            categories = False

        # MAKE THE HEATMAP VISUALIZATION
        plt.figure(figsize=figsize)
        sns.heatmap(cf, fmt="", annot=True, cmap=cmap, cbar=False, xticklabels=categories,
                    yticklabels=categories)

        if xyplotlabels:
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

        if title:
            plt.title(title)
        plt.savefig(self.fig_path / f'confusion_matrix.png')
        plt.show()

