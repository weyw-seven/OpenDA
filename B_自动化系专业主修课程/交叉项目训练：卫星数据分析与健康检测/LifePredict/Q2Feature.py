import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso
from utils import load_clean_data, draw_actual_vs_predicted
from sklearn.model_selection import train_test_split

# ========================== 模型配置函数 ==========================
CONFIG = {
    "model": "XGBoost",  # 可选值: "RandomForest", "XGBoost", "Ridge", "Lasso"
    "RandomForest": {
        "n_estimators": 50,
        "random_state": 42
    },
    "XGBoost": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 8,
        "random_state": 42
    },
    "Ridge": {
        "alpha": 0.01
    },
    "Lasso": {
        "alpha": 0.1
    }
}

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# ========================== 特征提取函数 ==========================
def helperGetFeatures(batch):
    """
    从电池数据中提取特征：
    - Q_100_10 统计量（方差、最小值）
    - 容量衰减曲线的斜率与截距
    - 放电容量、充电时间、内阻等
    """
    N = len(batch)
    y = np.zeros(N)
    DeltaQ_var = np.zeros(N)
    DeltaQ_min = np.zeros(N)
    CapFadeCycle2Slope = np.zeros(N)
    CapFadeCycle2Intercept = np.zeros(N)
    Qd2 = np.zeros(N)
    AvgChargeTime = np.zeros(N)
    MinIR = np.zeros(N)
    IRDiff2And100 = np.zeros(N)

    for i, key in enumerate(batch.keys()):
        y[i] = batch[key]['cycle_life'].item()
        DeltaQ = batch[key]['cycles']['99']['Qdlin'] - batch[key]['cycles']['9']['Qdlin']
        DeltaQ_var[i] = np.log10(np.abs(np.var(DeltaQ)))
        DeltaQ_min[i] = np.log10(np.abs(np.min(DeltaQ)))
        coeff2 = np.polyfit(batch[key]['summary']['cycle'][1:100], batch[key]['summary']['QD'][1:100], 1)
        CapFadeCycle2Slope[i] = coeff2[0]
        CapFadeCycle2Intercept[i] = coeff2[1]
        Qd2[i] = batch[key]['summary']['QD'][1]
        AvgChargeTime[i] = np.mean(batch[key]['summary']['chargetime'][1:6])
        temp = batch[key]['summary']['IR'][1:100]
        MinIR[i] = np.min(temp[temp != 0])
        IRDiff2And100[i] = batch[key]['summary']['IR'][99] - batch[key]['summary']['IR'][1]

    xTable = pd.DataFrame({
        'DeltaQ_var': DeltaQ_var,
        'DeltaQ_min': DeltaQ_min,
        'CapFadeCycle2Slope': CapFadeCycle2Slope,
        'CapFadeCycle2Intercept': CapFadeCycle2Intercept,
        'Qd2': Qd2,
        'AvgChargeTime': AvgChargeTime,
        'MinIR': MinIR,
        'IRDiff2And100': IRDiff2And100
    })

    return xTable, y


# ========================== 数据集划分 ==========================
def split_indices(bat_dict):
    """
    将数据划分为训练集、验证集和测试集。
    """
    keys_array = list(bat_dict.keys())
    numBat1 = len([key for key in keys_array if key.startswith('b1')])
    numBat2 = len([key for key in keys_array if key.startswith('b2')])

    val_ind = [keys_array[i] for i in np.hstack((np.arange(0, (numBat1 + numBat2), 2), 83))]
    train_ind = [keys_array[i] for i in np.arange(1, (numBat1 + numBat2 - 1), 2)]
    test_ind = [keys_array[i] for i in np.arange(numBat1 + numBat2, len(keys_array))]

    return train_ind, val_ind, test_ind


# ========================== 模型 ==========================
class ModelSelector:
    def __init__(self, config):
        """
        初始化模型选择器
        :param config: 配置字典，包含模型类型和对应参数
        """
        self.config = config
        self.model = self._select_model()

    def _select_model(self):
        """
        根据配置文件选择并初始化模型
        """
        model_name = self.config["model"]

        if model_name == "RandomForest":
            model = RandomForestRegressor(**self.config["RandomForest"])

        elif model_name == "XGBoost":
            model = XGBRegressor(**self.config["XGBoost"])

        elif model_name == "Ridge":
            model = Ridge(**self.config["Ridge"])

        elif model_name == "Lasso":
            model = Lasso(**self.config["Lasso"])

        else:
            raise ValueError("Invalid model name in CONFIG. Choose from 'RandomForest', 'XGBoost', 'Ridge', 'Lasso'.")

        return model

    def fit(self, X_train, y_train):
        """
        拟合模型
        :param X_train: 训练数据特征
        :param y_train: 训练数据标签
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        使用模型进行预测
        :param X: 输入特征数据
        :return: 预测结果
        """
        return self.model.predict(X)

    def get_model(self):
        """
        获取内部模型实例
        :return: 模型实例
        """
        return self.model


# ========================== 主函数 ==========================
def main():
    # 加载和预处理数据
    bat_dict = load_clean_data()
    train_ind, val_ind, test_ind = split_indices(bat_dict)

    # 提取特征和标签
    train_x, train_y = helperGetFeatures({key: bat_dict[key] for key in train_ind})
    val_x, val_y = helperGetFeatures({key: bat_dict[key] for key in val_ind})
    test_x, test_y = helperGetFeatures({key: bat_dict[key] for key in test_ind})

    train_x, X_temp, train_y, y_temp = train_test_split(train_x ,train_y, test_size=0.66, random_state=42)
    val_x, test_x, val_y, test_y = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    # 使用 ModelSelector 初始化模型
    model_selector = ModelSelector(CONFIG)
    
    # 模型训练
    model_selector.fit(train_x, train_y)

    # 预测
    val_predictions = model_selector.predict(val_x)
    test_predictions = model_selector.predict(test_x)

    # 计算验证集和测试集的指标
    print(f"Validation MSE: {mean_squared_error(val_y, val_predictions):.4f}")
    print(f"Validation MPE: {mean_absolute_percentage_error(val_y, val_predictions):.4f}")
    print(f"Test MSE: {mean_squared_error(test_y, test_predictions):.4f}")
    print(f"Test MPE: {mean_absolute_percentage_error(test_y, test_predictions):.4f}")

    # 绘制实际值与预测值
    draw_actual_vs_predicted(test_y, test_predictions)



# ========================== 程序入口 ==========================
if __name__ == "__main__":
    main()