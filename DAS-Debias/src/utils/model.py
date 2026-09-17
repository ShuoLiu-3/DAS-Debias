from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import seaborn as sns
import logging

logger = logging.getLogger(name='logger')


class DNN(nn.Module):
    def __init__(self, input_dim=4096, class_num=3):
        super(DNN, self).__init__()
        hidden_dim = 1024

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, class_num)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# [New Class] 轻量化中和模块
class NeutralizationAdapter(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=512):
        super(NeutralizationAdapter, self).__init__()
        # 瓶颈结构设计，降低参数量
        self.down_proj = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_dim, input_dim)

        # 初始化为近似恒等映射 (Identity mapping)，确保初始状态下不对模型造成破坏
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        residual = self.down_proj(x)
        residual = self.act(residual)
        residual = self.up_proj(residual)
        # 学习的是扰动 delta，所以最终结果是 x + delta
        return x + residual


def train_dnn(X_train, Y_train, device, num_epochs=20, input_dim=4096, class_num=3):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DNN(input_dim=input_dim, class_num=class_num).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_history = []
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # loss = criterion(outputs, torch.argmax(labels, dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_history.append(loss.item())
        # logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model, loss_history


def evaluate_dnn(model, X_val, Y_val, device, layer=0):
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)
    dataset_val = TensorDataset(X_val, Y_val)
    test_loader = DataLoader(dataset_val, batch_size=32, shuffle=False)
    model.eval()
    all_preds = []
    all_labels = []
    total_kl_div = 0
    total_mse = 0
    total_samples = 0
    val_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            _, true_labels = torch.max(labels, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())

            softmax_outputs = F.softmax(outputs, dim=1)
            kl_div = F.kl_div(F.log_softmax(outputs, dim=1), labels, reduction='batchmean')
            mse = F.mse_loss(softmax_outputs, labels)
            total_kl_div += kl_div.item() * inputs.size(0)
            total_mse += mse.item() * inputs.size(0)
            total_samples += inputs.size(0)

    val_loss /= len(test_loader)

    accuracy = sum([p == t for p, t in zip(all_preds, all_labels)]) / len(all_labels) * 100
    f1 = f1_score(all_labels, all_preds, average='macro')
    avg_kl_div = total_kl_div / total_samples
    avg_mse = total_mse / total_samples

    # Compute precision, recall, and F1 for class 0 and class 1
    precision_0 = precision_score(all_labels, all_preds, labels=[0], average='macro', zero_division=0)
    recall_0 = recall_score(all_labels, all_preds, labels=[0], average='macro', zero_division=0)
    f1_0 = f1_score(all_labels, all_preds, labels=[0], average='macro', zero_division=0)

    precision_1 = precision_score(all_labels, all_preds, labels=[1], average='macro', zero_division=0)
    recall_1 = recall_score(all_labels, all_preds, labels=[1], average='macro', zero_division=0)
    f1_1 = f1_score(all_labels, all_preds, labels=[1], average='macro', zero_division=0)
    # logger.info(f'layer {layer} Class 0 - Precision: {precision_0:.2f}, Recall: {recall_0:.2f}, F1: {f1_0:.2f}')
    # logger.info(f'layer {layer} Class 1 - Precision: {precision_1:.2f}, Recall: {recall_1:.2f}, F1: {f1_1:.2f}')

    cm = confusion_matrix(all_labels, all_preds)

    # logger.info(f'layer {layer} Accuracy: {accuracy:.2f}%, F1: {f1:.2f}, Avg KL Div: {avg_kl_div:.4f}, Avg MSE: {avg_mse:.4f}, Val Loss: {val_loss:.4f}')
    return accuracy, f1, avg_kl_div, avg_mse, cm


def plot_confusion_matrix(cm, name='confusion_matrix.png'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(name)
    plt.close()