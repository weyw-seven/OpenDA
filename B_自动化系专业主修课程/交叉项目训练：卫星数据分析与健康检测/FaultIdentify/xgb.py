import logging
from xgboost import XGBClassifier
from utils import *
import os

'''

助教仅需修改 path 和 Subsystem 即可完成测评

'''
CONFIG = {
    "data_path": "../data/",  # 数据路径，需包含 train.csv 和 test.csv 文件
    "Subsystem": "供配电",  # 分系统名称
    "result_path": "../result/" , # 结果保存路径
    "model": "XGBoost"  # 使用的模型名称:XGBoost
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
        """
        训练XGBoost模型。

        Args:
            X_train (ndarray): 训练集特征。
            y_train (ndarray): 训练集标签。

        Returns:
            XGBClassifier: 训练完成的模型。
        """
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
        model = self.train_model(X_train, y_train-1)
        y_pred, y_prob = predict(model, X_test)
        y_pred = y_pred + 1  # 恢复标签范围
        # 模型评估
        accuracy, report, matrix = evaluate_model(y_test, y_pred)
        logging.info(f"Overall Model accuracy: {accuracy:.4f}")
        logging.info(f"Overall Classification report:\n{report}")

        plot_confusion_matrix(matrix)
        # 保存结果
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