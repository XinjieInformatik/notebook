# Mechanisms of Action (MoA) Prediction
2020.12.01  public 47 / private 150
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

| index | strategy                                 | CV       | public  | private |
| ----- | ---------------------------------------- | -------- | ------- | ------- |
| 3-1   | avg. blend 2-1 2-3                       | 0.015516 | 0.01823 | 0.01615 |
| 3-2   | 2-1 2-3 target-wise lowest CV            | 0.015476 | 0.01825 | 0.01619 |
| 3-3   | target-wise weight blend 1-1 1-3 1-4 1-5 | 0.014219 | 0.01818 | 0.01613 |
| 3-4   | merge 3-2 3-3 weight blend               | 0.015353 | 0.01818 | 0.01612 |


### lesson learn
- 建立更合理的CV很重要 [drug CV split](https://www.kaggle.com/c/lish-moa/discussion/195195)
- train, public, private 的分布很重要[example](https://www.kaggle.com/c/lish-moa/discussion/200832)。比赛初期做target分析，留意举办方对split方式的申明，如果大概率是随机split的，可以相信CV，public test 当作另外一个fold做参考。[MoA](https://www.kaggle.com/c/lish-moa/overview)这个比赛有点特殊，public test 是人工划分出来的，让人不知道private test的划分方式，因此最后有些结论是相反的。
- 尝试过的方法，一开始无效的方法，甚至有效的方法要有记录，最后阶段要再试试[label smooth](https://www.kaggle.com/c/lish-moa/discussion/201729)。
- simple is good. 研究public kernel的时候，尝试简化他们的方案看看能不能取得同样的效果，再在这个基础上改进，改进不要太贪心，不要一味增加模型复杂度.

## 其他选手的有效方案
总结下来，online augmentation，不同结构的模型之后做blending，是主要的涨分点。
### 分析test set 与 train set 目标分布的差异
很有价值，推测test与train目标分布的差异，一来可以指导我们做CV,二来告诉我们CV有多可靠，三来可以用来post process 模型的输出
https://www.kaggle.com/cdeotte/moa-post-process-lb-1777

### lesson learn
- 模型之间的差异性很重要，一开始应该尝试更不同的方式，然后blending。即使单一模型的准确率不高，仍有价值
- 尽量保持模型简单，没有足够的收益，没必要一味增加seed，增加复杂度
- norm 既可以 col-wise 也可以 row-wise，也可以用神经网络去做[layer norm](https://www.kaggle.com/c/lish-moa/discussion/201051)
- 仔细阅读比赛的评分指标，包括loss计算时上下界的设定，建立更准确的local CV
- sklearn.decomposition 中，PCA, FactorAnalysis结果较为接近，FastICA与他们不同
- sklearn.preprocessing 中，不同的标准化 https://www.jianshu.com/p/580688e4a069. QuantileTransformer 受离群值影响小，但是特征间距离失真。注意norm用于行缩放到单位范数，standardization 用于列

### feature engineering 思路
- 生成 多项式特征
- Clip(-2, 2).round(1) 限制范围，限制噪声
- 以cp_time, cp_dose为group取gene和cell的mean，用原始feature减去作为新特征，注意要保留原始特征
- upsample where positive samples < 10
- PCA for cell and gene differently
- SVD for cell and gene differently

### online augmentation
mixup, swap, ctl增强等，人为引入噪声，引入更多数据量
train1[genes+cells].values + ctl1[genes+cells].values - ctl2[genes+cells].values
no_ctl_samples+ctl_sample1-ctl_sample2 to augment. ( ctl_sample1/2 are randomly choosed from all ctl samples ).

### pseudo labeling
使用 pseudo labeling 要小心，如果 public test 与 private test 差异较大，那 pseudo labeling可能会把自己坑了。如果 test 上预测的准确率不高，也没啥用。
https://www.kaggle.com/cdeotte/pseudo-labeling-qda-0-969
1. 在train上训练一个模型，预测test的label
2. 将train，test纵向拼接，作为新的数据集
3. 在拓展的新数据集上，重新训练一个模型，或者fine tuning 原来模型

### stacking
基模型出来的preds+原始features再训练模型，然后stacking出来的模型可以再和原始模型融合
[不同stacking间还可以融合](https://www.kaggle.com/c/lish-moa/discussion/204685)
[介绍stacking](https://mlwave.com/kaggle-ensembling-guide/)

### tabular to image
[1d-cnn](https://www.kaggle.com/c/lish-moa/discussion/202256)
[t-SNE to image](https://www.kaggle.com/c/lish-moa/discussion/195378)

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

### mixup in tabular data
https://www.kaggle.com/c/lish-moa/discussion/200702

### post-processing
- 以drug在train上做聚类，同时在test上做聚类，对于有高置信度的样本，赋予和相应drug同样的MoA
- 建立模型去预测drug，然后test上drug赋予相同的MoA

post-processing 很危险，除非有足够的准确率和收益,最后private最好还是选一个不用post的
reference：
https://www.kaggle.com/c/lish-moa/discussion/200596
https://www.kaggle.com/c/lish-moa/discussion/200609


### misc
- pretrain 再 finetune
