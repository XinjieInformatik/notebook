# AUTOML

值得关注的几个 auto tuning 框架

[auto-sklearn](https://github.com/automl/auto-sklearn)
[optuna](https://github.com/optuna/optuna#integrations)
[TPOT](https://github.com/EpistasisLab/tpot)
[hyperopt](https://github.com/hyperopt/hyperopt)


## optuna

目前看来， optuna 各种框架都支持（xgb, lgb, torch, tensorflow）值得尝试

- [optuna 官网](https://optuna.org)
- [optuna 文档](https://optuna.readthedocs.io/en/stable/index.html)
- [quickstart](https://colab.research.google.com/github/optuna/optuna/blob/master/examples/quickstart.ipynb#scrollTo=NGfNPIQHxo7i)
- [optuna 论文](https://dl.acm.org/doi/pdf/10.1145/3292500.3330701)
- [各个框架的 examples](https://github.com/optuna/optuna#integrations)
- [kaggle example](https://www.kaggle.com/corochann/optuna-tutorial-for-hyperparameter-optimization)
- [Distributed Optimization](https://optuna.readthedocs.io/en/v2.2.0/tutorial/004_distributed.html)

### optuna samplers
- [可选择的不同 samplers](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html?highlight=Distributed%20Optimization#sampling-algorithms)
- [TPESampler 原理](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html#optuna-samplers-tpesampler)

### optuna pruning methods
- [不同的 pruning 方法](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html?highlight=Distributed%20Optimization#pruning-algorithms)
- [MedianPruner](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html#optuna-pruners-medianpruner)

### 活动图
![20210116_170926_27](assets/20210116_170926_27.png)

### 贝叶斯优化
Grid Search和Randomized Search虽然可以让整个调参过程自动化，但它们无法从之前的调参结果中获取信息，可能会尝试很多无效的参数空间。而贝叶斯优化，会对上一次的评估结果进行追踪，建立一个概率模型，反应超参数在目标函数上表现的概率分布，即P(score|hyperparameters)，可用于指导下一次的参数选择。


### examples
#### xgb
```python
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna


def objective(trial):
    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.2, 100),

    }
    if param['booster'] == 'gbtree' or param['booster'] == 'gblinear':
        param['max_depth'] = trial.suggest_int('max_depth', 1, 9)
        param['eta'] = trial.suggest_float('eta', 1e-8, 1.0, log=True)
        param['gamma'] = trial.suggest_float('gamma', 1e-8, 1.0, log=True)
        param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
    if param['booster'] == 'dart':
        param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
        param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
        param['rate_drop'] = trial.suggest_float('rate_drop', 1e-8, 1.0, log=True)
        param['skip_drop'] = trial.suggest_float('skip_drop', 1e-8, 1.0, log=True)
    # XGBoostPruningCallback, stop the unpromising trial.
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'validation-auc')
    bst = xgb.train(param, dtrain, num_boost_round=trial.suggest_int('num_boost_round', 20, 1000),
                    evals=[(dvalid, 'validation')], early_stopping_rounds=20, callbacks=[pruning_callback])
    preds = bst.predict(dvalid, ntree_limit=bst.best_ntree_limit)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    return accuracy       


if __name__ == '__main__':
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction='maximize')
    study.optimize(objective, n_trials=100)
    trial = study.best_trial
    print(trial)
    print(trial.params)
```

#### lgb
```python
import lightgbm as lgb
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import optuna


def objective(trial):
    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    train_x, valid_x, train_y, valid_y = train_test_split(data, target ,test_size=0.25)
    dtrain = lgb.Dataset(train_x, label=train_y)
    dvalid = lgb.Dataset(valid_x, label=valid_y)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('baggign_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_chlid_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'auc')
    gbm = lgb.train(params, dtrain, valid_sets=[dvalid], verbose_eval=False, callbacks=[pruning_callback])
    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    return accuracy


if __name__ == '__main__':
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction='maximize')
    study.optimize(objective, n_trials=100)
    trial = study.best_trial
    print(trial.params)
```

#### pytorch
```python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import optuna

DEVICE = torch.device('cuda')
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 10
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10

def define_model(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layers = []
    in_features = 28 * 28
    for i in range(n_layers):
        out_features = trial.suggest_int('n_units_l{}'.format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float('dropout_l{}'.format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))
        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))
    return nn.Sequential(*layers)

def get_mnist():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DIR, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DIR, train=False, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    return train_loader, valid_loader

def objective(trial):
    model = define_model(trial).to(DEVICE)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    train_loader, valid_loader = get_mnist()
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break
            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print("Accuracy: {}".format(trial.value))
    print("Best params: {}".format(trial.params))

    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_slice(study)
    optuna.visualization.plot_contour(study. params=['n_estimators', 'max_depth'])
    optuna.visualization.plot_param_importances(study)
    study.trials_dataframe()
```
