import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def plot_confusion_matrix(matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, cmap='Blues', fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()



def analyze_all_faults(y_test, y_pred, fault_map_file, CONFIG):
    """
    对每种故障类型进行分析并保存结果到CSV。

    Args:
        y_test (ndarray): 测试集真实标签。
        y_pred (ndarray): 模型预测标签。
        fault_map_file (str): JSON文件路径，包含标签到故障名称的映射。
        CONFIG (dict): 配置文件，包含保存路径信息。

    Returns:
        None
    """
    with open(fault_map_file, encoding="utf-8") as f:
        fault_map = json.load(f)

    results = []
    total_samples = len(y_test)
    correct_predictions = np.sum(y_test == y_pred)
    overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    for fault_label in np.unique(y_test):
        indices = y_test == fault_label
        total = np.sum(indices)
        correct = np.sum(y_pred[indices] == fault_label)
        accuracy = correct / total if total > 0 else 0
        results.append({
            "Fault Label": fault_label,
            "Description": fault_map.get(str(fault_label), f"Unknown({fault_label})"),
            "Total": total,
            "Correct": correct,
            "Accuracy": f"{accuracy:.6f}"
        })

    # 添加总计行
    results.append({
        "Fault Label": "Total",
        "Description": "Overall",
        "Total": total_samples,
        "Correct": correct_predictions,
        "Accuracy": f"{overall_accuracy:.6f}"
    })

    df = pd.DataFrame(results)
    csv_filename = os.path.join(CONFIG["result_path"], CONFIG["Subsystem"], f"{CONFIG['Subsystem']}_Identify_Result.csv")

    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)  # 确保目录存在
    df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
    logging.info(f"Fault analysis results saved to {csv_filename}")

def load_data(file_path):
    """
    加载CSV数据文件。

    Args:
        file_path (str): 文件路径。

    Returns:
        pd.DataFrame: 加载后的数据。
    """
    data = pd.read_csv(file_path)
    logging.info(f"Successfully loaded data from {file_path}")
    return data

def filter_abnormal_data(data, label_col='label'):
    """
    仅保留异常数据（label 非 0）。

    Args:
        data (pd.DataFrame): 数据集。
        label_col (str): 标签列名称。

    Returns:
        pd.DataFrame: 仅包含异常数据的子集。
    """
    filtered_data = data[data[label_col] != 0]
    logging.info(f"Filtered abnormal data: {filtered_data.shape[0]} samples")
    return filtered_data

def extract_features(data, label_col='label'):
    """
    提取特征部分，去除标签列。

    Args:
        data (pd.DataFrame): 数据集。
        label_col (str): 标签列名称，默认 'label'。

    Returns:
        tuple: 特征部分 (X) 和标签部分 (y)。
    """
    return data.drop(columns=[label_col]), data[label_col]

def predict(model, X_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return y_pred, y_prob

def evaluate_model(y_test, y_pred):
    """
    评估分类模型性能，计算准确率、分类报告和混淆矩阵。

    Args:
        y_test (ndarray): 测试集真实标签。
        y_pred (ndarray): 模型预测标签。

    Returns:
        tuple: 准确率、分类报告和混淆矩阵。
    """
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    return accuracy, report, matrix
