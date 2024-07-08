import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
import matplotlib.pyplot as plt
import torch_training_utils.data_utils as data_utils
import models
from tqdm import tqdm


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    y_true, y_pred = [], []

    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)

        loss = criterion(outputs, targets.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        true = targets.argmax(1)
        pred = outputs.argmax(1)

        y_true.extend(true.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_score = f1_score(y_true, y_pred, average='weighted')
    return train_loss, train_score


def validate(model, data_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)

            loss = criterion(outputs, targets.float())
            val_loss += loss.item()

            true = targets.argmax(1)
            pred = outputs.argmax(1)

            y_true.extend(true.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    val_loss /= len(data_loader)
    val_score = f1_score(y_true, y_pred, average='weighted')
    return val_loss, val_score


def test(model, data_loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)

            true = targets.argmax(1)
            pred = outputs.argmax(1)

            y_true.extend(true.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    test_report = classification_report(y_true, y_pred, digits=4)
    test_score = f1_score(y_true, y_pred, average='weighted')
    return test_report, test_score


def predict(model, data_loader, criterion, device):
    model.eval()
    y_pred = []
    avg_loss = 0.0

    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)

            loss = criterion(outputs, targets.float())
            avg_loss += loss.item()

            pred = outputs.argmax(1)
            y_pred.extend(pred.cpu().tolist())

    avg_loss /= len(data_loader)
    return avg_loss, y_pred


# 绘制训练、验证曲线
def plot_learning_curves(train_losses, train_scores, val_losses, val_scores, vlines_x=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Learning Curves')
    x = range(len(train_losses))
    ax1.plot(x, train_losses, label='Train')
    ax1.plot(x, val_losses, label='Validation')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epochs')
    ax2.plot(x, train_scores, label='Train')
    ax2.plot(x, val_scores, label='Validation')
    ax2.set_ylabel('F1-score')
    ax2.set_xlabel('Epochs')
    plt.legend(loc="lower right")

    if vlines_x:
        ax1.axvline(x=vlines_x, color='red', linestyle='dashed')
        ax2.axvline(x=vlines_x, color='red', linestyle='dashed')

    plt.show()


# 根据类别比例提取类别权重
def get_class_weights(labels):
    _, counts = np.unique(labels, return_counts=True)
    weights = np.sum(counts) / counts
    return torch.FloatTensor(weights)


# k折交叉验证
def nn_kfold_cv(n_splits, dataset, model, lr, batch_size, epochs, device):
    # 定义交叉验证损失与分数
    cv_losses = []
    cv_scores = []

    # 计算标签权重（针对标签不平衡）
    X = dataset.get_np_data()
    y = dataset.get_1d_labels()
    class_weights = get_class_weights(y).to(device)

    # 开始交叉验证数据分割
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"Fold {fold + 1}")
        # 定义模型，需要特别设置数据中的通道数和信号长度
        model_optim = copy.deepcopy(model)  # 复制初始模型，防止重复训练
        criterion = nn.CrossEntropyLoss(weight=class_weights)  # 加类别权重损失函数
        optimizer = torch.optim.Adam(model_optim.parameters(), lr=lr)

        # 记录当前fold训练验证损失和分数
        train_losses, train_scores = [], []
        val_losses, val_scores = [], []

        # 取当前fold数据
        train_loader = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=train_idx)
        val_loader = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=val_idx)
        for epoch in tqdm(range(epochs)):
            train_loss, train_score = train(model_optim, train_loader, optimizer, criterion, device)  # 模型训练
            val_loss, val_score = validate(model_optim, val_loader, criterion, device)  # 模型验证

            # 记录过程中的训练、验证损失和分类指标
            train_losses.append(train_loss)
            train_scores.append(train_score)
            val_losses.append(val_loss)
            val_scores.append(val_score)

        # 取最后一轮训练分数作为该fold最终结果
        cv_losses.append(val_losses[-1])
        cv_scores.append(val_scores[-1])

        # 绘制学习曲线
        plot_learning_curves(train_losses, train_scores, val_losses, val_scores)

    print('K fold validation losses: ', cv_losses)
    print('K fold validation f1-scores: ', cv_scores)
    print('Mean validation loss: ', np.mean(cv_losses))
    print('Mean validation score: ', np.mean(cv_scores))
    return np.mean(cv_losses), np.mean(cv_scores)


# 留一交叉验证
def nn_loo_cv(dataset, model, lr, batch_size, epochs, device):
    # 提取真标签
    y_true = dataset.get_1d_labels()
    # 计算标签权重（针对标签不平衡）
    class_weights = get_class_weights(y_true).to(device)

    # 记录验证损失与验证损失最低点
    val_losses, min_val_indices = [], []
    val_preds = []  # 留一交叉验证过程中的验证数据预测标签

    # 遍历所有数据，使每个数据作为一次验证集
    for val_idx in tqdm(range(len(dataset))):
        # 除验证集之外的数据作为训练数据
        train_idx = list(range(len(dataset))).remove(val_idx)

        model_optim = copy.deepcopy(model)  # 复制初始模型，防止重复训练
        criterion = nn.CrossEntropyLoss(weight=class_weights)  # 加类别权重损失函数
        optimizer = torch.optim.Adam(model_optim.parameters(), lr=lr)

        # 取当前fold数据
        train_loader = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=train_idx)
        val_loader = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=[val_idx])

        cv_val_loss = []
        for epoch in range(epochs):
            _, _ = train(model_optim, train_loader, optimizer, criterion, device)  # 模型训练
            epoch_val_loss, _ = validate(model_optim, val_loader, criterion, device)  # 模型验证
            cv_val_loss.append(epoch_val_loss)  # 添加每个epoch的验证损失

        val_loss, val_pred = predict(model_optim, val_loader, criterion, device)  # 使用训练后的模型预测验证集标签
        # 记录过程中的验证损失、分类结果和验证损失最低的epoch
        val_losses.append(val_loss)
        val_preds += val_pred
        min_val_indices.append(np.argmin(cv_val_loss))

    # 计算验证指标
    val_scores = f1_score(y_true, val_preds, average='weighted')
    return (np.mean(val_losses), np.mean(val_scores), np.array(min_val_indices),
            classification_report(y_true, val_preds, digits=4))


def nn_regular_training(X, y, model, lr, batch_size, epochs, model_save_name, device, split_ratio=[.6, .2, .2], patience=10):
    # 计算标签权重（针对标签不平衡）
    class_weights = get_class_weights(y).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 装载数据
    train_loader, val_loader, test_loader = data_utils.get_loaders(X, y, batch_size, split_ratio=split_ratio)

    # 记录损失与分类分数
    best_val_loss = np.Inf
    best_val_score = 0
    counter = 0
    train_losses, train_scores = [], []
    val_losses, val_scores = [], []

    for epoch in range(epochs):
        train_loss, train_score = train(model, train_loader, optimizer, criterion, device)  # 模型训练
        val_loss, val_score = validate(model, val_loader, criterion, device)  # 模型验证
        print("EPOCH: {}/{}".format(epoch + 1, epochs))
        print("Train loss: {:.6f}, Train F1: {:.4f}".format(
            train_loss, train_score))
        print("Val loss: {:.6f}, Val F1: {:.4f}\n".format(
            val_loss, val_score))

        # 记录过程中的训练、验证损失和分类指标
        train_losses.append(train_loss)
        train_scores.append(train_score)
        val_losses.append(val_loss)
        val_scores.append(val_score)

        # 检查是否提前停止
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_score = val_score
            counter = 0
            # 验证损失达到新低点时保存模型
            torch.save(model.state_dict(), model_save_name)
        else:
            counter += 1

        if counter >= patience:
            print(f'Early stopping after {epoch + 1} epochs')
            break

    # 绘制学习曲线
    plot_learning_curves(train_losses, train_scores, val_losses, val_scores)
    print(f'Saved model validation score: {best_val_score}')

    # 读取训练过程中保存的最优模型，在测试集上检验模型表现
    model.load_state_dict(torch.load(model_save_name))
    test_report, test_score = test(model, test_loader, device)  # 模型测试
    print('Model result on test set')
    print(test_report)
    print(f'Test score: {test_score}')
    return best_val_loss, best_val_score, test_score, test_report
