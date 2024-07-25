import os
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (f1_score, classification_report,
                             roc_auc_score, roc_curve, auc, RocCurveDisplay)
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
import matplotlib.pyplot as plt
import torch_training_utils.data_utils as data_utils
from torch_training_utils.utils import set_device, seed_everything
from tqdm import tqdm
from itertools import cycle
from scipy import interp


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
    train_score = f1_score(y_true, y_pred, average='macro')
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
    val_score = f1_score(y_true, y_pred, average='macro')
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
    test_score = f1_score(y_true, y_pred, average='macro')
    return test_report, test_score


def predict(model, data_loader, criterion, device, return_prob=False):
    model.eval()
    y_pred = []
    avg_loss = 0.0

    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)

            loss = criterion(outputs, targets.float())
            avg_loss += loss.item()

            if not return_prob:
                pred = outputs.argmax(1)
                y_pred.extend(pred.cpu().tolist())
            else:
                y_pred.extend(outputs.cpu().tolist())

    avg_loss /= len(data_loader)
    return avg_loss, y_pred


# 绘制训练、验证曲线
def plot_learning_curves(train_losses, train_scores, val_losses, val_scores, mark_minimum=True):
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

    # 红色虚线标记验证损失最低点
    if mark_minimum:
        min_loss_pos = np.argmin(train_losses)
        ax1.axvline(x=min_loss_pos, color='red', linestyle='dashed')
        ax2.axvline(x=min_loss_pos, color='red', linestyle='dashed')

    plt.show()


# 绘制ROC曲线
def plot_roc_curves(y_true, y_score, save_name=None):
    n_classes = y_true.shape[1]

    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig, ax = plt.subplots(figsize=(6, 6))

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.4f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.4f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_true[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for class {class_id}",
            color=color,
            ax=ax,
            plot_chance_level=(class_id == 2),
        )

    _ = ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )

    if save_name:
        save_path = './plots/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(save_path + save_name + '.png')


# 根据类别比例提取类别权重
def get_class_weights(labels):
    _, counts = np.unique(labels, return_counts=True)
    weights = np.sum(counts) / counts
    return torch.FloatTensor(weights)


# k折交叉验证
def nn_kfold_cv(n_splits, dataset, model, lr, batch_size, epochs):
    seed_everything()
    device = set_device()
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
        model_optim.to(device)
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
def nn_loo_cv(dataset, model, lr, batch_size, epochs, plot_name=None):
    seed_everything()
    device = set_device()
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
        model_optim.to(device)
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

        val_loss, val_pred = predict(model_optim, val_loader, criterion, device, return_prob=True)  # 使用训练后的模型预测验证集标签
        # 记录过程中的验证损失、分类结果和验证损失最低的epoch
        val_losses.append(val_loss)
        val_preds.extend(val_pred)
        min_val_indices.append(np.argmin(cv_val_loss))

    one_hot_labels = np.array(dataset.labels)
    val_preds = np.array(val_preds)
    # 计算验证指标
    val_scores = roc_auc_score(one_hot_labels, val_preds, multi_class='ovr')
    if plot_name:
        plot_roc_curves(one_hot_labels, val_preds, save_name=plot_name)
    return (np.mean(val_losses), np.mean(val_scores), np.array(min_val_indices),
            classification_report(y_true, np.argmax(val_preds, axis=1), digits=4))


def nn_regular_training(X, y, model, lr, batch_size, epochs, model_save_name,
                        split_ratio=[.6, .2, .2], patience=10, verbose=0):
    seed_everything()
    device = set_device()
    # 计算标签权重（针对标签不平衡）
    class_weights = get_class_weights(y).to(device)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 装载数据
    train_loader, val_loader, test_loader = data_utils.get_loaders(
        X, y, batch_size, split_ratio=split_ratio, verbose=verbose)

    # 记录损失与分类分数
    best_val_loss = np.Inf
    best_val_score = 0
    counter = 0
    train_losses, train_scores = [], []
    val_losses, val_scores = [], []

    for epoch in range(epochs):
        train_loss, train_score = train(model, train_loader, optimizer, criterion, device)  # 模型训练
        val_loss, val_score = validate(model, val_loader, criterion, device)  # 模型验证

        if verbose:
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
            if verbose:
                print(f'Early stopping after {epoch + 1} epochs')
            break

    # 读取训练过程中保存的最优模型，在测试集上检验模型表现
    model.load_state_dict(torch.load(model_save_name))
    test_report, test_score = test(model, test_loader, device)  # 模型测试

    if verbose:
        print('Model result on test set')
        print(test_report)
        print(f'Test score: {test_score}')
        # 绘制学习曲线
        plot_learning_curves(train_losses, train_scores, val_losses, val_scores)
        print(f'Saved model validation score: {best_val_score}')

    return best_val_loss, best_val_score, test_score, test_report


def nn_train_test(X, y, model, lr, batch_size, epochs, split_ratio=[.7, .3], plot_name=None, verbose=0):
    seed_everything()
    device = set_device()
    # 计算标签权重（针对标签不平衡）
    class_weights = get_class_weights(y).to(device)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 装载数据
    train_loader, test_loader = data_utils.get_train_test_loaders(
        X, y, batch_size, split_ratio=split_ratio, verbose=verbose)

    train_losses, train_scores = [], []
    for epoch in range(epochs):
        train_loss, train_score = train(model, train_loader, optimizer, criterion, device)  # 模型训练

        if verbose:
            print("EPOCH: {}/{}".format(epoch + 1, epochs))
            print("Train loss: {:.6f}, Train F1: {:.4f}".format(
                train_loss, train_score))

        # 记录过程中的训练、验证损失和分类指标
        train_losses.append(train_loss)
        train_scores.append(train_score)

    test_report, _ = test(model, test_loader, device)  # 模型测试
    test_loss, test_preds = predict(model, test_loader, criterion, device, return_prob=True)

    # 计算AUC指标
    test_labels = np.array(y)[test_loader.dataset.indices]
    one_hot_labels = np.eye(len(np.unique(y)))[test_labels]
    test_preds = np.array(test_preds)
    test_auc = roc_auc_score(one_hot_labels, test_preds, multi_class='ovr')
    if plot_name:
        plot_roc_curves(one_hot_labels, test_preds, save_name=plot_name)

    if verbose:
        print('Model result on test set')
        print(test_report)
        print(f'Test score: {test_auc}')

    return test_loss, test_auc, test_report
