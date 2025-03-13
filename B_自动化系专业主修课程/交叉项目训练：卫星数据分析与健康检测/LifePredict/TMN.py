import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from utils import *
CONFIG = {
    "num_cycles": 100,
    "num_points": 10,
    "batch_size": 16,
    "num_epochs": 1000,
    "learning_rate": 0.001,
    "kernel_size": 3,
    "inchannels": 3,
    "save_path": "./best_model.pth"
}

device = torch.device('cuda:4')



# ----------------------- 模型定义模块 -----------------------

def dct(x, norm='ortho'):
    """
    实现一维 DCT (Discrete Cosine Transform)
    :param x: 输入数据, [B, H, W]
    :param norm: 正交归一化
    :return: DCT 结果
    """
    N = x.shape[-1]
    v = torch.arange(N, device=x.device).float()
    coeff = torch.cos(torch.pi * (v + 0.5).view(1, -1) * v.view(-1, 1) / N)
    if norm == 'ortho':
        coeff[0] *= 1 / torch.sqrt(torch.tensor(2.0, device=x.device))
        coeff *= torch.sqrt(torch.tensor(2.0 / N, device=x.device)) 
    return torch.matmul(x, coeff)


class TimesBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=2):
        super(TimesBlock, self).__init__()
        self.k = k

        # 时域特征提取模块
        self.inception = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.dct_mapper = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.dct_bn = nn.BatchNorm2d(out_channels)

        # 注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 残差通道映射，确保输入和输出通道一致
        self.channel_mapper = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.channel_bn = nn.BatchNorm2d(out_channels)

        # 特征融合模块
        self.fc = nn.Conv2d(out_channels * 2, out_channels, kernel_size=(1, 1))
        self.fc_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x  # 保存输入
        B, C, num_cycles, num_points = x.shape

        # 输入数据标准化
        x = torch.nan_to_num(x, nan=0.0, posinf=1, neginf=-1)
        mean = x.mean(dim=(-1, -2), keepdim=True)
        std = x.std(dim=(-1, -2), keepdim=True) + 1e-6
        x_norm = (x - mean) / std

        # DCT 提取频域特征
        x_reshaped = x_norm.view(B * C, num_cycles, num_points)
        dct_features = dct(x_reshaped)[:, :, :self.k].view(B, C, num_cycles, self.k)
        dct_features = torch.nn.functional.interpolate(
            dct_features, size=(num_cycles, num_points), mode='bilinear', align_corners=False
        )

        # 频域与时域特征提取
        dct_features = self.dct_bn(self.dct_mapper(dct_features))
        x_incept = self.inception(x)

        # 特征融合
        x_combined = torch.cat([x_incept, dct_features], dim=1)
        x_out = self.fc_bn(self.fc(x_combined))

        # 残差连接
        residual_mapped = self.channel_bn(self.channel_mapper(residual))  # 映射通道

        return x_out + residual_mapped



class tmn(nn.Module):
    def __init__(self, input_channels, hidden_dim, num_cycles, num_points, k=1):
        super(tmn, self).__init__()
        self.timesblock1 = TimesBlock(input_channels, hidden_dim, k)
        self.timesblock2 = TimesBlock(hidden_dim, hidden_dim, k)
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 8, kernel_size=(3, 3), padding=1),  # 新增
            nn.BatchNorm2d(hidden_dim // 8),  # 新增
            nn.ReLU()
        )



        # 全局平均池化层
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim // 8, 64),
            nn.LeakyReLU (),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

        self.num_cycles = num_cycles
        self.num_points = num_points

    def forward(self, x):
        B, C, TN = x.shape

        # 重塑输入 [B, C, num_cycles, num_points]
        x = x.view(B, C, self.num_cycles, -1)

        x = self.timesblock1(x)  # [B, hidden_dim, num_cycles, num_points]
        x = self.timesblock2(x)

        x = self.conv(x)

        x = self.global_pool(x)

        return self.fc(x)



# ----------------------- 训练与评估模块 -----------------------
def train_and_evaluate(model, train_loader, val_loader, test_loader, num_epochs, lr):
    """
    训练模型并进行评估，保留验证集最优权重。
    :param model: PyTorch 模型
    :param train_loader: 训练数据 DataLoader
    :param val_loader: 验证数据 DataLoader
    :param test_loader: 测试数据 DataLoader
    :param num_epochs: 最大训练 epoch 数
    :param lr: 学习率
    :return: 测试集的 targets 和 predictions
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 记录训练和验证过程
    train_losses, val_losses = [], []
    best_train_loss = float('inf')  # 初始化最优验证损失
    best_model_state = None      # 保存最优模型状态

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
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                val_loss += criterion(outputs.squeeze(), y_val).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # 打印每个 epoch 的训练和验证损失
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}")

        # 检查验证集是否是最优
        if train_losses[-1]< best_train_loss:
            best_train_loss = train_losses[-1] 
            best_model_state = model.state_dict()  # 保存模型最优权重

    # 加载验证集最优权重
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with Loss")

    # 测试阶段
    predictions, targets = [], []
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            outputs = model(X.to(device)).cpu().numpy().flatten()
            predictions.extend(outputs)
            targets.extend(y.numpy().flatten())

    # 打印测试结果
    print(f"Test MSE: {mean_squared_error(targets, predictions):.4f}")
    print(f"Test MPE: {mean_absolute_percentage_error(targets, predictions):.4f}")

    return targets, predictions

def create_loader(X, y):
    return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32),
                                    torch.tensor(y, dtype=torch.float32)),
                        batch_size=CONFIG["batch_size"], shuffle=True)

# ----------------------- 主函数 -----------------------
def main():
    batch_data = load_clean_data()
    X, y = prepare_battery_data(batch_data, num_cycles=CONFIG["num_cycles"], num_points=CONFIG["num_points"])
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.66 ,random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    
    train_loader = create_loader(X_train, y_train)
    val_loader = create_loader(X_val, y_val)
    test_loader = create_loader(X_test, y_test)

    

    model = tmn(input_channels=CONFIG['inchannels'], hidden_dim=32, num_cycles=CONFIG["num_cycles"],num_points=CONFIG["num_points"]  ).to(device)
    targets, predictions = train_and_evaluate(model, train_loader, val_loader, test_loader,
                                                                        CONFIG["num_epochs"], CONFIG["learning_rate"])

    draw_actual_vs_predicted (targets, predictions, savepath='fig/tmn')

if __name__ == "__main__":
    main()
