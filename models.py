import torch.nn as nn


# CNN网络结构
class CNN(nn.Module):
    def __init__(self, in_channels, sig_length, n_classes, hidden_channels=16, kernel_size=3):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding='same'),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding='same'),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.AvgPool1d(2),
        )
        flat_dim = sig_length // 2 * hidden_channels
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)
        return x
