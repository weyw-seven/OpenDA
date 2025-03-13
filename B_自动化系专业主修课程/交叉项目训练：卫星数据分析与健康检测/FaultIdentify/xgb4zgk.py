import numpy as np
import logging
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

import os
from utils import *

'''

助教仅需修改 path 和 Subsystem 即可完成测评

'''
CONFIG = {
    "data_path": "../data/",  # 数据路径，需包含 train.csv 和 test.csv 文件
    "Subsystem": "姿轨控",  # 分系统名称
    "result_path": "../result/" , # 结果保存路径
    "model": "XGBoost"  # 使用的模型名称:XGBoost/LightGBM
}


# 配置日志输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FaultJudge:
    """
    故障检测类，封装了数据加载、模型训练、预测以及评估的完整流程。
    """
    def __init__(self,CONFIG):
        self.model = CONFIG["model"]


    def train_model(self, X_train, y_train):
        if self.model == "XGBoost":
            model = XGBClassifier(n_estimators=300,
                 learning_rate=0.1918, 
                 max_depth=4, 
                 random_state=42)
        elif self.model == "LightGBM":
            model = LGBMClassifier(n_estimators=300,
                 learning_rate=0.1918, 
                 max_depth=4, 
                 random_state=42)

        logging.info(f"Start training model {self.model} ")
        model.fit(X_train, y_train)
        logging.info("Model training complete.")
        return model

#--------------请注意一定要替换成负Y翼A轴故障和陀螺2输出常零故障的标签----------------
    def separate_learning(self, train_data, test_data):
        """
        针对label=3和label=4的数据单独学习并优化。

        Args:
            train_data (pd.DataFrame): 训练数据。
            test_data (pd.DataFrame): 测试数据。

        Returns:
            tuple: 预测值和概率，适用于label=3和label=4。
        """
        target_labels = [3, 4]

        # 筛选目标类别的数据
        train_subset = train_data[train_data['label'].isin(target_labels)].copy()
        test_subset = test_data[test_data['label'].isin(target_labels)].copy()

        # 重置标签范围（从0开始）
        train_subset['label'] = train_subset['label'] - target_labels[0]
        test_subset['label'] = test_subset['label'] - target_labels[0]

        # 提取特征和标签
        X_train, y_train = extract_features(train_subset)
        X_test, y_test = extract_features(test_subset)

        # 模型训练
        logging.info("Starting separate learning for labels 3 and 4...")
        # model = SVC(kernel='linear', C=10000, gamma='scale', probability=True)
        model = LogisticRegression(C=1, max_iter=50000)
        model.fit(X_train, y_train)


        # 模型预测
        y_pred = model.predict(X_test)
        y_pred_adjusted = y_pred + target_labels[0]  # 恢复标签为3和4
        y_prob = model.predict_proba(X_test)

        return y_pred_adjusted, y_prob
    

    def run(self, train_file, test_file, fault_map_file,CONFIG):
        """
        运行完整流程，包括数据加载、模型训练、预测和评估。

        Args:
            train_file (str): 训练集文件路径。
            test_file (str): 测试集文件路径。
            fault_map_file (str): JSON文件路径，包含标签到故障名称的映射。

        Returns:
            tuple: 包括准确率、分类报告和混淆矩阵。
        """
        logging.info("Starting the fault detection process...")

        # 加载数据
        train_data = load_data(train_file)
        test_data = load_data(test_file)

        # 筛选异常数据
        train_data = filter_abnormal_data(train_data)
        test_data = filter_abnormal_data(test_data)

        # 数据处理
        X_train, y_train = extract_features(train_data)
        X_test, y_test = extract_features(test_data)

        # 模型训练与预测（整体）
        model = self.train_model(X_train, y_train - 1)
        y_pred, y_prob = predict(model, X_test)

        # 恢复标签范围
        y_pred = np.array(y_pred).flatten() + 1

        # 单独学习 label=3 和 label=4
        y_pred_3_4, _ = self.separate_learning(train_data, test_data)

        # 替换原预测结果中的 label=3 和 label=4 部分
        mask_3_4 = (y_test == 3) | (y_test == 4)
        mask_3_4 = mask_3_4.to_numpy()  # 转换为 NumPy 数组

        # 确保布尔索引结果和替换值都是一维数组
        y_pred[mask_3_4] = y_pred_3_4.flatten()

        # 模型评估
        accuracy, report, matrix = evaluate_model(y_test, y_pred)
        logging.info(f"Overall Model accuracy: {accuracy:.4f}")
        logging.info(f"Overall Classification report:\n{report}")

        plot_confusion_matrix(matrix)
        # 故障分析
        analyze_all_faults(y_test, y_pred, fault_map_file,CONFIG)

        return accuracy, report, matrix


if __name__ == '__main__':
    # 使用CONFIG中的路径配置
    train_file = os.path.join(CONFIG['data_path'], CONFIG['Subsystem'], 'train.csv')
    test_file = os.path.join(CONFIG['data_path'], CONFIG['Subsystem'], 'test.csv')
    fault_map_file = os.path.join(CONFIG['data_path'], CONFIG['Subsystem'], 'itoa.json')

    # 创建FaultJudge实例并运行
    fault_judge = FaultJudge(CONFIG)
    accuracy, report, matrix = fault_judge.run(train_file, test_file, fault_map_file, CONFIG)
    