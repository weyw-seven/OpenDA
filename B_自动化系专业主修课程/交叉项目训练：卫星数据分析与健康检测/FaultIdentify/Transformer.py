import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import *
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# 配置日志输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 配置文件，包含模型参数和文件路径信息
CONFIG = {
    "data_path": "../data/",  # 数据路径，需包含 train.csv 和 test.csv 文件
    "Subsystem": "供配电",    # 分系统名称
    "result_path": "../result/",  # 结果保存路径
}

class TransformerClassifier(nn.Module):
    """
    基于 Transformer 的分类模型。
    """
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=3, hidden_dim=64):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        """
        # x: [batch_size, seq_len, input_dim] -> [batch_size, seq_len, hidden_dim]
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        # 最后做线性分类
        x = self.fc(x)
        return x

class trJudge:
    """
    故障检测类，封装了基于 Transformer 模型的训练、预测和评估流程。
    """

    def train_transformer(
        self, 
        X_train, 
        y_train, 
        num_classes,
        batch_size=32,
        lr=1e-4,
        max_epoch=100
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = X_train.shape[1]

        model = TransformerClassifier(input_dim=input_dim, num_classes=num_classes).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_train_tensor = X_train_tensor.unsqueeze(1)  # [n_samples, 1, input_dim]

        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(max_epoch):
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # 若需要，可开启梯度截断，防止梯度爆炸
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            logging.info(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

        return model


    def run(self, train_file, test_file, label_col='label'):
        logging.info("Starting the fault detection process...")
        train_data = load_data(train_file)
        test_data = load_data(test_file)
        train_data = filter_abnormal_data(train_data, label_col=label_col)
        test_data = filter_abnormal_data(test_data, label_col=label_col)

        # 提取特征和标签
        X_train, y_train = extract_features(train_data, label_col=label_col)
        X_test, y_test = extract_features(test_data, label_col=label_col)

        if X_train.isna().sum().sum() > 0 or X_test.isna().sum().sum() > 0:
            logging.warning("Data contains NaNs. Consider cleaning or imputing.")
        # 简单去掉含有NaN的样本（或可做插值等）
        X_train = X_train.dropna()
        X_test = X_test.dropna()

        y_train = y_train[X_train.index]
        y_test = y_test[X_test.index]


        unique_labels = np.unique(y_train)
        logging.info(f"Unique train labels (before shift): {unique_labels}")
        # 假设都从1开始，shift到0开始
        y_train -= 1
        y_test -= 1

        # 再次确认训练标签是否从0开始连续
        unique_labels = np.unique(y_train)
        logging.info(f"Unique train labels (after shift): {unique_labels}")


        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 确定类别数量
        num_classes = len(np.unique(y_train))

        # 训练模型
        model = self.train_transformer(
            X_train_scaled, 
            y_train.values, 
            num_classes=num_classes,
            batch_size=64,     
            lr=1e-4,            
            max_epoch=200      
        )

        # 测试阶段
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1).to(device)
            outputs = model(X_test_tensor)
            y_pred = torch.argmax(outputs, axis=1).cpu().numpy()


        y_pred += 1
        y_test += 1

        accuracy, report, matrix = self.evaluate_model(y_test, y_pred)
        logging.info(f"Overall Model accuracy: {accuracy:.4f}")
        logging.info(f"Overall Classification report:\n{report}")

        plot_confusion_matrix(matrix)

        return accuracy, report, matrix

if __name__ == '__main__':
    # 使用 CONFIG 中的路径配置
    train_file = os.path.join(CONFIG['data_path'], CONFIG['Subsystem'], 'train.csv')
    test_file = os.path.join(CONFIG['data_path'], CONFIG['Subsystem'], 'test.csv')

    # 创建 FaultJudge 实例并运行
    fault_judge = trJudge()
    accuracy, report, matrix = fault_judge.run(train_file, test_file, label_col='label')

    # 输出结果
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", matrix)