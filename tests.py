import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch_training_utils.utils import seed_everything, configure_plot_parameters
from torch_training_utils.ClassificationTrainer import ClassificationTrainer
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification


seed_everything(42)
configure_plot_parameters()

n_classes = 2
n_features = 100
hidden_size = 128

X, y = make_classification(n_samples=1000, n_features=n_features, n_classes=n_classes)
X = torch.tensor(X, dtype=torch.float32)
y = nn.functional.one_hot(torch.tensor(y, dtype=torch.long))

# 准备数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

model = torch.nn.Sequential(
    torch.nn.Linear(in_features=n_features, out_features=hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=hidden_size, out_features=n_classes),
)

# 训练参数
lr = 1e-3
batch_size = 64
epochs = 100

# 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

save_paths = {
    'log_path': Path.cwd() / f'test_output/log/',  # 训练过程log保存路径
    'model_path': Path.cwd() / f'test_output/model/',  # 模型保存路径
    'fig_path': Path.cwd() / f'test_output/figs/',  # 结果图保存路径
}

# 创建保存路径文件夹
for key, value in save_paths.items():
    value.mkdir(parents=True, exist_ok=True)

# 初始化Trainer
trainer = ClassificationTrainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    metric=f1_score,
    metric_params={
        'average': 'macro',
        'zero_division': 0.0,
    },
    save_paths=save_paths
)
# 装载数据
trainer.load_data(train_dataset, test_dataset, val_dataset=val_dataset, batch_size=batch_size)
# 训练模型
trainer.fit_holdout(epochs=epochs, patience=epochs)
# 绘制学习曲线
trainer.plot_learning_curves(metric_name='F1 score')


