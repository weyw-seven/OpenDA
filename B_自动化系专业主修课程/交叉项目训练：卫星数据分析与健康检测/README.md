# 卫星数据分析和健康监测大作业项目文档

## 项目简介
该项目旨在通过分析卫星数据，完成 **故障检测**、**故障诊断** 和 **寿命预测** 三大任务。项目涉及数据处理、机器学习模型开发以及结果分析，并结合卫星实际应用场景设计优化方案。

---

## 项目组织架构

```
Project
├── data
│   ├── bettery
│   │   ├── batch1.pkl
│   │   ├── batch2.pkl
│   │   └── batch3.pkl
│   ├── 供配电
│   │   ├── all.csv
│   │   ├── atoi.json
│   │   ├── itoa.json
│   │   ├── test.csv
│   │   └── train.csv
│   ├── 姿轨控
│   │   ├── all.csv
│   │   ├── atoi.json
│   │   ├── itoa.json
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── 姿轨控_report.csv
│   └── 激光载荷
│       ├── all.csv
│       ├── atoi.json
│       ├── itoa.json
│       ├── test.csv
│       ├── train.csv
│       └── 激光载荷_report.csv
├── FaultIdentify
│   ├── Transformer.py
│   ├── utils.py
│   ├── xgb.py
│   └── xgb4zgk.py
├── FaultDetect
│   ├── utils.py
│   └── xgb.py
├── LifePredict
│   ├── 1DCNN.py
│   ├── Q2Feature.py
│   ├── TMN.py
│   └── utils.py
├── result
│   ├── 供配电
│   │   └── Fault_Detect_Result_供配电.csv
│   ├── 姿轨控
│   │   └── Fault_Detect_Result_姿轨控.csv
│   └── 激光载荷
│       └── Fault_Detect_Result_激光载荷.csv
├── process.ipynb
├── README.md
```

---

## 功能模块说明

### 1. 数据存储模块（`data/`）
- 包含项目所需的所有数据集，按子系统分类：
  - **供配电**：存储供配电相关数据及 JSON 映射文件。
  - **姿轨控**：存储姿轨控相关数据及分析报告。
  - **激光载荷**：存储激光载荷相关数据及分析报告。
  - **bettery**：存储电池相关的批量数据（`pkl` 格式）。

### 2. 故障识别模块（`FaultIdentify/`）
- 包含用于故障检测和诊断的模型代码。
  - **`Transformer.py`**：基于 Transformer 的故障识别方法。
  - **`xgb.py` 和 `xgb4zgk.py`**：基于 XGBoost 的故障分类代码。
  - **`utils.py`**：通用工具函数。

### 3. 故障判断模块（`FaultDetect/`）
- 实现了对数据的判别与模型训练：
  - **`xgb.py`**：XGBoost 判别逻辑。
  - **`utils.py`**：通用工具函数。

### 4. 寿命预测模块（`LifePredict/`）
- 包含电池寿命预测的相关实现：
  - **`1DCNN.py`**：端到端预测模型。
  - **`Q2Feature.py`**：从统计量提取特征后做预测（根据助教的Demo）。
  - **`TMN.py`**：使用DCT的时间序列建模方法。
  - **`utils.py`**：通用工具函数。

### 5. 结果存储模块（`result/`）
- 存储各子系统的故障检测和诊断结果：
  - **供配电**：`FaultIdentify_Result_供配电.csv`
  - **姿轨控**：`FaultIdentify_Result_姿轨控.csv`
  - **激光载荷**：`FaultIdentify_Result_激光载荷.csv`

### 6. 其他文件
- **`process.ipynb`**：数据处理和分析的 Jupyter Notebook。
- **`README.md`**：项目说明文档。

---

## 使用说明

### 环境配置
- **Python 版本**：3.9
- 安装依赖库：
  ```bash
  pip install -r requirements.txt
  ```

### 运行流程
1. **数据预处理**：
   - 调用主路径下的 `preprocess.ipynb/`文件对待测数据进行预处理，按照前文所述的进入`data/` 文件夹下
   - 确保 `data/` 文件夹下的数据完整。
2. **运行模型**：
   - 进入对应模块（如 `FaultIdentify/`）。
   - 执行相应的 Python 脚本，例如：
     ```bash
     python xgb.py
     ```
3. **查看结果**：
   - 故障检测的结果将存储在 `result/` 文件夹下对应的子系统目录中。
   - 选做部分的结果会直接print出来

4. **拓展接口**：
   - 预留了模型改变的接口，可以直接修改所使用的moded
   - 选做部分给出了CONFIG在文件开头，可以调节参数观察实验结果
---

## 项目特点
1. **高完成度**：涵盖供配电、姿轨控、激光载荷等多个子系统的故障检测与诊断；同时完成了选做任务；

2. **多模型对比**：尝试了多种经典与前沿的机器学习模型（如 XGBoost、Transformer）。
3. **灵活扩展性**：代码结构非常清晰，便于添加新模型或分析方法。
4. **创新**：引入了DCT来对频域信息进行捕捉来完成寿命预测任务。
---

## 贡献者
- **学生**：张博仕

---


