
'''
工具函数模块：
可视化函数
数据加载与处理函数

'''
import matplotlib.pyplot as plt
import pickle
import numpy as np

# ========================== 可视化函数 ==========================
def draw_actual_vs_predicted(y_true, y_pred, savepath='fig/actual_vs_predicted.png'):
    """
    绘制实际值与预测值的散点图，并标出每个点到理想对角线的垂直虚线。
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, c='blue', alpha=0.5, label='Predicted vs Actual')

    for i in range(len(y_true)):
        plt.plot([y_true[i], y_true[i]], [y_pred[i], y_true[i]], linestyle='--', color='red', linewidth=0.8)

    plt.plot([0, 2000], [0, 2000], linestyle='--', color='green', label='Ideal Line')
    plt.xlim(0, 2000)
    plt.ylim(0, 2000)
    plt.xlabel("Actual Cycle Life")
    plt.ylabel("Predicted Cycle Life")
    plt.legend()
    plt.tight_layout()
    plt.title("Actual vs Predicted Cycle Life")
    plt.savefig(savepath)
    plt.show()

def plot_losses(train_losses, val_losses, test_losses, num_epochs, save_path='fig/loss_curve.png'):
    """
    绘制训练、验证和测试的损失曲线。
    :param train_losses: 训练损失列表
    :param val_losses: 验证损失列表
    :param test_losses: 测试损失列表
    :param num_epochs: 训练总 Epoch 数
    :param save_path: 图像保存路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='tab:blue', linestyle='-')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss', color='tab:orange', linestyle='--')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', color='tab:green', linestyle='-.')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training, Validation, and Test Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)

def plot_metrics(val_mse_list, val_mpe_list, test_mse_list, test_mpe_list, num_epochs, save_path='fig/metrics_curve.png'):
    """
    绘制验证集和测试集的 MSE 与 MPE 曲线。
    :param val_mse_list: 验证集 MSE 列表
    :param val_mpe_list: 验证集 MPE 列表
    :param test_mse_list: 测试集 MSE 列表
    :param test_mpe_list: 测试集 MPE 列表
    :param num_epochs: 训练总 Epoch 数
    :param save_path: 图像保存路径
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左轴：MSE 曲线
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE', color='tab:blue')
    ax1.plot(range(1, num_epochs + 1), val_mse_list, label='Val MSE', color='tab:cyan', linestyle='-')
    ax1.plot(range(1, num_epochs + 1), test_mse_list, label='Test MSE', color='tab:blue', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # 右轴：MPE 曲线
    ax2 = ax1.twinx()
    ax2.set_ylabel('MPE', color='tab:red')
    ax2.plot(range(1, num_epochs + 1), val_mpe_list, label='Val MPE', color='tab:red', linestyle='-.')
    ax2.plot(range(1, num_epochs + 1), test_mpe_list, label='Test MPE', color='tab:pink', linestyle=':')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # 添加图例与标题
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.85))
    plt.title('Validation and Test MSE/MPE')
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)

# ----------------------- 数据加载与处理 -----------------------
def load_clean_data():
    batch1 = pickle.load(open(r'../data/bettery/batch1.pkl', 'rb'))
    batch2 = pickle.load(open(r'../data/bettery/batch2.pkl', 'rb'))
    batch3 = pickle.load(open(r'../data/bettery/batch3.pkl', 'rb'))
    for invalid in ['b1c8', 'b1c10', 'b1c12', 'b1c13', 'b1c22']:
        batch1.pop(invalid, None)
    for invalid in ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']:
        batch2.pop(invalid, None)
    for invalid in ['b3c37', 'b3c2', 'b3c23', 'b3c32', 'b3c42', 'b3c43']:
        batch3.pop(invalid, None)
    return {**batch1, **batch2, **batch3}

def prepare_battery_data(batch, num_cycles=100, num_points=500):
    X_data, y_data = [], []
    for key in batch.keys():
        y_data.append(batch[key]['cycle_life'].item())
        battery_data = []
        for cycle in range(1, num_cycles + 1):
            cycle_str = str(cycle)
            if cycle_str in batch[key]['cycles']:
                cycle_data = batch[key]['cycles'][cycle_str]
                features = []
                for feature in ['V', 'I', 'T']:
                    original_data = np.array(cycle_data[feature])
                    sampled_data = np.interp(
                        np.linspace(0, len(original_data) - 1, num_points),
                        np.arange(len(original_data)),
                        original_data
                    )
                    features.append(sampled_data)
                battery_data.append(features)
        X_data.append(np.concatenate(battery_data, axis=-1))  # [channels, num_cycles * num_points]
    return np.array(X_data), np.array(y_data)
