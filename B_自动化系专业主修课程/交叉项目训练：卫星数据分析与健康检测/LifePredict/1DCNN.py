import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from utils import *

# 设备配置
device = torch.device('cuda')

# 超参数的配置字典
CONFIG = {
    "num_cycles": 100,
    "num_points": 10,
    "batch_size": 16,
    "num_epochs": 1000,
    "learning_rate": 0.002,
    "dilations": [1, 3, 9, 27, 81],
    "num_channels": 32,
    "kernel_size": 3
}

# ----------------------- 模型定义模块 -----------------------
class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=(kernel_size - 1) * dilation // 2,  # 保证输出长度一致
            dilation=dilation
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(self.relu(self.conv(x)))

class BatteryRULCNN(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size=3, dilations=[1, 3, 9, 27]):
        super(BatteryRULCNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # 多层卷积，每一层有不同的膨胀率
        for dilation in dilations:
            self.layers.append(
                Conv1DBlock(
                    in_channels=input_channels if dilation == 1 else num_channels, 
                    out_channels=num_channels,
                    kernel_size=kernel_size, 
                    dilation=dilation
                )
            )
        
        self.global_pooling = nn.AdaptiveAvgPool1d(1)  # 全局平均池化，输出维度固定
        self.flatten = nn.Flatten()

        # 全连接层，动态匹配输入大小
        self.fc = nn.Sequential(
            nn.Linear(num_channels, 256),  # 输入大小自动匹配池化后的输出
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # 输入形状: [batch_size, channels, seq_len]
        for layer in self.layers:
            x = layer(x)
        
        x = self.global_pooling(x)  # 全局平均池化 -> [batch_size, channels, 1]
        x = self.flatten(x)         # 展平成 [batch_size, channels]
        out = self.fc(x)            # 全连接层输出
        return out


# ----------------------- 训练与评估模块 -----------------------
def train_and_evaluate(model, train_loader, val_loader, test_loader, num_epochs, lr):
    """训练模型并进行评估，记录每个 Epoch 的损失和指标。"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, val_losses, test_losses = [], [], []  # 训练和验证损失
    val_mse_list, val_mpe_list = [], []  # 验证集 MSE 和 MPE
    tes_mse_list, tes_mpe_list = [], []  # 测试集 MSE 和 MPE


    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        # 验证阶段
        model.eval()
        val_loss = 0
        val_targets, val_predictions = [], []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                val_loss += criterion(outputs.squeeze(), y_val).item()
                val_predictions.extend(outputs.cpu().numpy().flatten())
                val_targets.extend(y_val.cpu().numpy().flatten())
        val_losses.append(val_loss / len(val_loader))
        
        test_loss = 0
        test_predictions, test_targets = [], []  # 测试集预测和真实值
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(device), y_test.to(device)
                outputs = model(x_test)
                test_loss += criterion(outputs.squeeze(), y_test).item()
                test_predictions.extend(outputs.cpu().numpy().flatten())
                test_targets.extend(y_test.cpu().numpy().flatten())
        test_losses.append(test_loss / len(test_loader))

        # 计算 MSE 和 MPE
        mse = mean_squared_error(val_targets, val_predictions)
        mpe = mean_absolute_percentage_error(val_targets, val_predictions)
        val_mse_list.append(mse)
        val_mpe_list.append(mpe)

        mse = mean_squared_error(test_targets, test_predictions)
        mpe = mean_absolute_percentage_error(test_targets, test_predictions)
        tes_mse_list.append(mse)
        tes_mpe_list.append(mpe)
        

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_losses[-1]:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}, "
              f"Val MSE: {mse:.4f}, Val MPE: {mpe:.4f}")

    return train_losses, val_losses, test_losses, val_mse_list, val_mpe_list, tes_mse_list, tes_mpe_list

# 数据加载器
def create_loader(X, y):
    return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32),
                                    torch.tensor(y, dtype=torch.float32)),
                        batch_size=CONFIG["batch_size"], shuffle=True)

# ----------------------- 主函数 -----------------------
def main():
    # 数据加载与处理
    batch_data = load_clean_data()
    X, y = prepare_battery_data(batch_data, CONFIG["num_cycles"], CONFIG["num_points"])
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.66, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_loader = create_loader(X_train, y_train)
    val_loader = create_loader(X_val, y_val)
    test_loader = create_loader(X_test, y_test)

    model = BatteryRULCNN(input_channels=3, num_channels=CONFIG["num_channels"], 
                          kernel_size=CONFIG["kernel_size"], dilations=CONFIG["dilations"]).to(device)

    train_losses, val_losses, test_losses, val_mse_list, val_mpe_list, test_mse_list, test_mpe_list = train_and_evaluate(
        model, train_loader, val_loader, test_loader, CONFIG["num_epochs"], CONFIG["learning_rate"])

    plot_losses(train_losses, val_losses, test_losses, CONFIG["num_epochs"], save_path='fig/loss_curve.png')

    plot_metrics(val_mse_list, val_mpe_list, test_mse_list, test_mpe_list, CONFIG["num_epochs"], save_path='fig/metrics_curve.png')

if __name__ == "__main__":
    main()
