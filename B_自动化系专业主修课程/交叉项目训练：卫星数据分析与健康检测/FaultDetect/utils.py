"""
功能描述：
本文件提供了一系列用于评估分类模型性能的函数，包括混淆矩阵绘制、模型性能评估、
ROC 曲线绘制以及生成故障检测报告并保存为 CSV 文件。
"""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def plot_confusion_matrix(matrix):
    """
    绘制混淆矩阵的热力图。

    Args:
        matrix (ndarray): 混淆矩阵数据。

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Fault", "Fault"],
        yticklabels=["No Fault", "Fault"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


def evaluate_model(y_test, y_pred):
    """
    评估分类模型性能，计算准确率、分类报告和混淆矩阵。

    Args:
        y_test (list or ndarray): 测试集真实标签。
        y_pred (list or ndarray): 模型预测标签。

    Returns:
        tuple: 包括准确率 (accuracy)、分类报告 (report) 和混淆矩阵 (matrix)。
    """
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    return accuracy, report, matrix


def plot_roc_curve(y_test, y_prob):
    """
    绘制 ROC 曲线并计算 AUC 值。

    Args:
        y_test (list or ndarray): 测试集真实标签。
        y_prob (list or ndarray): 模型预测的正类概率值。

    Returns:
        None
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


def generate_report(y_test, y_pred, subsystem_name, output_path):
    """
    生成故障检测报告并保存为 CSV 文件。

    Args:
        y_test (list or ndarray): 测试集真实标签。
        y_pred (list or ndarray): 模型预测标签。
        subsystem_name (str): 分系统名称，例如 "姿轨控"。
        output_path (str): 输出文件路径。

    Returns:
        None
    """
    # 计算指标
    total_normal = sum(y_test == 0)
    total_fault = sum(y_test == 1)
    false_alarms = sum((y_pred == 1) & (y_test == 0))  # 虚警数
    missed_alarms = sum((y_pred == 0) & (y_test == 1))  # 漏警数
    true_alarms = sum((y_pred == 1) & (y_test == 1))    # 正确报警数

    # 计算虚警率和漏警率
    false_alarm_rate = false_alarms / total_normal if total_normal > 0 else 0
    miss_alarm_rate = missed_alarms / total_fault if total_fault > 0 else 0

    # 组织表格数据
    data = [
        {
            "分系统": subsystem_name,
            "名称": "正常数据",
            "检测次数": total_normal,
            "报警次数": false_alarms,
            "虚警率/漏警率": f"{false_alarm_rate:.6f}"
        },
        {
            "分系统": subsystem_name,
            "名称": "故障数据",
            "检测次数": total_fault,
            "报警次数": true_alarms,
            "虚警率/漏警率": f"{miss_alarm_rate:.6f}"
        }
    ]

    # 转为 DataFrame
    df = pd.DataFrame(data)

    # 保存 CSV 文件
    csv_file = os.path.join(output_path, f"{subsystem_name}_Detection_Result.csv")
    df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    print(f"Report saved to {csv_file}")
