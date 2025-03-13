import pandas as pd
import logging
from utils import *
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 配置日志输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 配置文件，包含模型参数和文件路径信息
# 助教仅需修改 path 和 Subsystem 即可完成测评
CONFIG = {
    "path": "../data/",  # 数据路径，需包含 train.csv 和 test.csv 文件
    "model": "XGBoost",  # 使用的模型名称
    "Subsystem": "姿轨控",  # 分系统名称
    "result_path": "../result/" , # 结果保存路径
}

class FaultJudge:
    """
    故障检测类，封装了数据加载、模型训练、预测以及评估的完整流程。
    """

    def __init__(self, n_estimators=730, learning_rate=0.1918, max_depth=6, random_state=42):
        """
        初始化故障检测模型及参数。

        Args:
            n_estimators (int): 
            learning_rate (float): 
            max_depth (int): 
            random_state (int): 
        """
        self.model_params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'random_state': random_state,
            'use_label_encoder': False,
            'eval_metric': 'logloss'  # 使用logloss作为评估指标
        }

    @staticmethod
    def load_data(file_path):
        data = pd.read_csv(file_path)
        logging.info(f"Successfully loaded data from {file_path}")
        return data

    @staticmethod
    def process_labels(data, label_col='label'):
        data[label_col] = data[label_col].apply(lambda x: 0 if x == 0 else 1)
        return data

    @staticmethod
    def extract_features(data, label_col='label'):
        return data.drop(label_col, axis=1)

    def train_model(self, train_data):
        """
        训练XGBoost模型。

        Args:
            train_data (pd.DataFrame): 训练数据集。

        Returns:
            XGBClassifier: 训练完成的模型。
        """
        train_data = self.process_labels(train_data)  # 处理标签
        X_train = self.extract_features(train_data)  # 提取特征
        y_train = train_data['label']  # 获取标签

        # 初始化并训练模型
        model = XGBClassifier(**self.model_params)
        model.fit(X_train, y_train)
        logging.info("Model training complete.")
        return model

    def predict(self, model, test_data):
        """
        使用模型进行预测。

        Args:
            model (XGBClassifier): 训练好的模型。
            test_data (pd.DataFrame): 测试数据集。

        Returns:
            tuple: 包括预测值 (y_pred)、真实值 (y_test) 和预测概率 (y_prob)。
        """
        test_data = self.process_labels(test_data)  # 处理标签
        X_test = self.extract_features(test_data)  # 提取特征
        y_test = test_data['label']  # 获取真实标签

        # 获取预测值和预测概率
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        return y_pred, y_test, y_prob

    def run(self, train_file, test_file):
        """
        运行完整流程，包括数据加载、模型训练、预测和评估。

        Args:
            train_file (str): 训练集文件路径。
            test_file (str): 测试集文件路径。

        Returns:
            tuple: 包括准确率、分类报告、混淆矩阵、预测值和真实值。
        """
        logging.info("Starting the fault detection process...")

        # 加载训练集和测试集数据
        train_data = self.load_data(train_file)
        test_data = self.load_data(test_file)

        # 模型训练与预测
        model = self.train_model(train_data)
        y_pred, y_test, y_prob = self.predict(model, test_data)

        # 模型评估
        accuracy, report, matrix = evaluate_model(y_test, y_pred)
        logging.info(f"Model accuracy: {accuracy:.4f}")
        logging.info(f"Classification report:\n{report}")

        # 绘制混淆矩阵和ROC曲线
        plot_confusion_matrix(matrix)
        plot_roc_curve(y_test, y_prob)

        return accuracy, report, matrix, y_pred, y_test


if __name__ == '__main__':
    # 使用CONFIG中的路径配置
    train_file = CONFIG['path'] + CONFIG['Subsystem'] + '/train.csv'
    test_file = CONFIG['path'] + CONFIG['Subsystem'] + '/test.csv'

    # 创建FaultJudge实例并运行
    fault_judge = FaultJudge()
    accuracy, report, matrix, y_pred, y_test = fault_judge.run(train_file, test_file)

    # 输出结果
    print("Accuracy:\n", accuracy)
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", matrix)

    # 生成故障检测报告并保存为CSV
    generate_report(y_test, y_pred, CONFIG['Subsystem'], CONFIG['result_path']+CONFIG['Subsystem'] + '/')