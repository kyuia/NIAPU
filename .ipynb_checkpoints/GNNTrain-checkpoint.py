from GraphSageModel import GNN7L_Sage
from Paths import PATH_TO_IMAGES, PATH_TO_REPORTS, PATH_TO_MODELS

import pandas as pd
import seaborn as sn
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, data, epochs, lr, weight_decay, classes, model_name):    
    data = data.to(device)
    model = model.to(device)

    # 统一模型命名规则（添加疾病ID和模式）
    title = f"{model_name}_{epochs}_{weight_decay:.4f}".replace('.', '_')
    model_path = os.path.join(PATH_TO_MODELS, title)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_mask = data['train_mask']
    val_mask = data['val_mask']
    labels = data.y
    output = ''

    # 训练曲线数据
    train_acc_curve = []
    train_lss_curve = []

    best_train_acc = 0
    best_val_acc = 0
    best_train_lss = 999
    best_loss_epoch = 0

    for e in tqdm(range(epochs+1)):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        output = logits.argmax(1)
        train_loss = F.nll_loss(logits[train_mask], labels[train_mask])
        train_acc = (output[train_mask] == labels[train_mask]).float().mean()
        train_loss.backward()
        optimizer.step()

        # 记录训练曲线
        train_acc_curve.append(train_acc.item())
        train_lss_curve.append(train_loss.item())

        # 更新最佳指标
        if train_acc > best_train_acc:
            best_train_acc = train_acc
        if train_loss < best_train_lss:
            best_train_lss = train_loss
            best_loss_epoch = e
            torch.save(model.state_dict(), model_path)  # 保存最佳模型

        # 验证集评估
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            output = logits.argmax(1)
            val_loss = F.nll_loss(logits[val_mask], labels[val_mask])
            val_acc = (output[val_mask] == labels[val_mask]).float().mean()

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # 定期打印日志
        if e % 20 == 0 or e == epochs:
            print(f'[Epoch: {e:04d}] '
                  f'train_loss: {train_loss.item():.4f}, '
                  f'train_acc: {train_acc.item():.4f}, '
                  f'val_loss: {val_loss.item():.4f}, '
                  f'val_acc: {val_acc.item():.4f}, '
                  f'best_train_acc: {best_train_acc.item():.4f}, '
                  f'best_val_acc: {best_val_acc.item():.4f}')

    # 绘制训练曲线
    plt.figure(figsize=(12,7))
    plt.plot(train_acc_curve)
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy')
    plt.savefig(os.path.join(PATH_TO_IMAGES, f'{title}_acc.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(12,7))
    plt.plot(train_lss_curve)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.savefig(os.path.join(PATH_TO_IMAGES, f'{title}_loss.png'), dpi=300)
    plt.close()

    return output

def predict_from_saved_model(
    disease_id: str,
    mode: str,
    data,
    classes,
    epochs: int = 40000,
    weight_decay: float = 0.0005,
    plot_results: bool = True,
    save_to_file: bool = True
):
    """加载指定模型进行预测"""
    data = data.to(device)
    labels = data.y
    # 动态生成模型路径
    model_name = f"GraphSAGE_{disease_id}_{mode}_{epochs}_{weight_decay:.4f}".replace('.', '_')
    model_path = os.path.join(PATH_TO_MODELS, model_name)
    # 调试输出
    print(f"[DEBUG] 正在加载模型: {model_name}")
    print(f"[DEBUG] 模型路径: {model_path}")    
    # 根据模式确定分类数
    num_classes = 2 if mode == 'binary' else 5
    loaded_model = GNN7L_Sage(data).to(device)
    
    try:
        loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        raise FileNotFoundError(f"模型文件 {model_path} 不存在，请检查路径或先运行训练脚本")
    
    loaded_model.eval()
    with torch.no_grad():
        logits = loaded_model(data.x, data.edge_index)
        output = logits.argmax(1)

    # 生成报告
    test_mask = data['test_mask']
    if plot_results:
        print(classification_report(
            labels[test_mask].to('cpu'),
            output[test_mask].to('cpu'),
            target_names=classes
        ))

    # 保存结果
    if save_to_file:
        report_name = f"{disease_id}_{mode}_report.csv"
        report_path = os.path.join(PATH_TO_REPORTS, report_name)
        class_report = classification_report(
            labels[test_mask].to('cpu'),
            output[test_mask].to('cpu'),
            output_dict=True,
            target_names=classes
        )
        pd.DataFrame(class_report).to_csv(report_path)

    # 混淆矩阵
    if plot_results:
        cm = confusion_matrix(
            labels[test_mask].to('cpu'),
            output[test_mask].to('cpu'),
            normalize='true'
        )
        plt.figure(figsize=(7,7))
        sn.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix ({mode})')
        plt.savefig(os.path.join(PATH_TO_IMAGES, f'{model_name}_cm.png'), dpi=300)
        plt.close()

    return output, logits, loaded_model