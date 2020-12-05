# Mechanisms of Action (MoA) Prediction
2020.12.01
https://www.kaggle.com/c/lish-moa/overview
- 特征：一个实验下，800+维度的gene特征与100维度cell特征
- 目标：输出206个MoA（药理作用）的置信度 - 预测当前药物该实验下，基因与细胞表现下是否有该药理作用 - 多任务分类问题

## 我的方案
因为数据量小，神经网络容易过拟合，而且目标稀疏，所以把神经网络当作随机森林用，每个单一模型使用5-10个fold stratification， 5个seed，每个seed的fold的shuffle规律不同。
在两个不同的分布下（不同的CV，stratification），分别训练4-6个NN模型，然后使用target-wise的方法做融合。

### 单纯stratification的CV下训练的5个NN

| index | model   | CV       | CV ord. | public  | pb ord. | private | pv ord. |
| ----- | ------- | -------- | ------- | ------- | ------- | ------- | ------- |
| 1-1   | nn_base | 0.014533 | 4       | 0.01834 | 2       | 0.01626 | 2       |
| 1-2   | resnet  | 0.014480 | 3       | 0.01852 | 5       | 0.01634 | 5       |
| 1-3   | nn_no   | 0.014282 | 1       | 0.01838 | 3       | 0.01632 | 4       |
| 1-4   | drug_id | 0.014369 | 2       | 0.01826 | 1       | 0.01623 | 1       |
| 1-5   | tabnet  | 0.014866 | 5       | 0.01839 | 4       | 0.01629 | 3       |

### 结合drug种类做的CV下训练的6个NN

| index | model      | CV       | CV ord. | public  | pb ord. | private | pv ord. |
| ----- | ---------- | -------- | ------- | ------- | ------- | ------- | ------- |
| 2-1   | drug_id    | 0.015626 | 2       | 0.01837 | 3       | 0.01627 | 3       |
| 2-2   | nn_v3      | 0.015620 | 1       | 0.01833 | 1       | 0.01624 | 2       |
| 2-3   | nn_no_drug | 0.015642 | 3       | 0.01833 | 1       | 0.01623 | 1       |
| 2-4   | resnet     | 0.015790 | 5       | 0.01862 | 6       | 0.01644 | 6       |
| 2-5   | tabnet     | 0.015685 | 6       | 0.01842 | 5       | 0.01633 | 5       |
| 2-6   | nn_base    | 0.015654 | 4       | 0.01840 | 4       | 0.01627 | 3       |

### 模型融合

| index | 策略                           | CV       | public  | private |
| ----- | ------------------------------ | -------- | ------- | ------- |
| 1     | 加权融合 2-1 2-3               | 0.015516 | 0.01615 | 0.01823 |
| 2     | 2-1 2-3 206-target中选取最小CV | 0.015476 | 0.01619 | 0.01825 |
|       |                                |          |         |         |

### lesson learn

## 其他选手的有效方案

### 分析test set 与 train set 目标分布的差异
很有价值，推测test与train目标分布的差异，一来可以指导我们做CV,二来告诉我们CV有多可靠，三来可以用来post process 模型的输出
https://www.kaggle.com/cdeotte/moa-post-process-lb-1777

### lesson learn
- 模型之间的差异性很重要，一开始应该尝试更不同的方式即使单一模型的准确率不高，然后blending
- 低分的模型，在blending中仍然是有价值的，必要一味降低其权重
- 尽量保持模型简单，没有足够的收益，没必要一味增加seed，增加复杂度

### feature engineering 思路
- 生成 多项式特征
- Clip(-2, 2).round(1) 限制范围，限制噪声
- 以cp_time, cp_dose为group取gene和cell的mean，用原始feature减去作为新特征，注意要保留原始特征
- upsample where positive samples < 10
- PCA for cell and gene differently
- SVD for cell and gene differently

### pesudo label, stacking, stacking+features

### stacking
stacking 出来的features+原始features再训练模型，然后stacking出来的模型可以再和原始模型融合


### multilabel to multiclass
把multitask转化为multilabel的问题，再融合这两个方式下训练出来的模型
https://www.kaggle.com/c/lish-moa/discussion/200992

### early stop by target
https://www.kaggle.com/c/lish-moa/discussion/200784

### layernorm
分组处理gene，cell，使用layernorm

https://www.kaggle.com/c/lish-moa/discussion/201051

### SVM/XGB
https://www.kaggle.com/c/lish-moa/discussion/200656

### post-processing
- 以drug在train上做聚类，同时在test上做聚类，对于有高置信度的样本，赋予和相应drug同样的MoA
- 建立模型去预测drug，然后test上drug赋予相同的MoA

post-processing 很危险，除非有足够的准确率和收益,最后private最好还是选一个不用post的
reference：
https://www.kaggle.com/c/lish-moa/discussion/200596
https://www.kaggle.com/c/lish-moa/discussion/200609

### mixup in tabular data
https://www.kaggle.com/c/lish-moa/discussion/200702

### misc
- pretrain 再 finetune
