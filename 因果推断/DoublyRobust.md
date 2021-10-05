# Doubly Robust


## one / two model
对于one/two model 因果效应的预估直接用不同treatment下的响应做差得到。
$$
\theta_{t}(X)=\mathbb{E}\left[Y^{(t)}-Y^{(0)} \mid X\right] = \mathbb{E}\left[g_{t}(X, T)-g_{0}(X, T) \mid X\right]
$$


## IPS
如果使用IPS:
$$
Y_{i, t}^{I P S}=\frac{Y_{i} \left\{T_{i}=t\right\}}{p_{t}\left(X_{i}, T_{i}\right)}
$$
推导一下，使用 $\mathbb{E}\left[Y_{i, t}^{I P S} \mid X, T\right]$是否是 $\mathbb{E}\left[Y_{i}^{(t)} \mid X_{i}, T_{i}\right]$ 的一种等效形式

$$
\begin{aligned}
\mathbb{E}\left[Y_{i, t}^{I P S} \mid X, T\right] &=\mathbb{E}\left[\frac{Y_{i} 1\left\{T_{i}=t\right\}}{p_{t}\left(X_{i}, T_{i}\right)} \mid X_{i}, T_{i}\right]=\mathbb{E}\left[\frac{Y_{i}^{(t)} 1\left\{T_{i}=t\right\}}{p_{t}\left(X_{i}, T_{i}\right)} \mid X_{i}, T_{i}\right] \\
&=\mathbb{E}\left[\frac{Y_{i}^{(t)} \mathbb{E}\left[1\left\{T_{i}=t\right\} \mid X_{i}, T_{i}\right]}{p_{i}\left(X_{i}, T_{i}\right)} \mid X_{i}, T_{i}\right]=\mathbb{E}\left[Y_{i}^{(t)} \mid X_{i}, T_{i}\right]
\end{aligned}
$$

$\theta_t(X) = Y_{i, t}^{I P S}-Y_{i, 0}^{I P S}$

## Doubly Robust Learning
- g是响应模型，使用原始的label作为y，$g_t(X_i, T_i)$ 表示treatment为T_i时，样本i的转化概率
- p为倾向分模型，使用treatment作为y，$p_t(X_i, T_i)$ 对treatment T_i的倾向分
- $Y_{i, t}^{D R}$ 为最终模型在treatment T_i下的y
- 注意多treatment时，$Y^{D R}, g, p$ 维度为 (N, T)，其中N为样本个数，T为treatment个数
- 用$\theta_t(X)$表示样本上的因果效应， $\theta_t(X) = Y_{i, t}^{D R}-Y_{i, 0}^{D R}$

双稳健方法的主要优点是最终估计值 $θ_t(X)$ 的均方误差仅受回归估计值 $g_t(X,W)$ 和倾向估计值 $p_t(X,W)$ 的均方误差乘积的影响。因此，只要其中之一是准确的，那么最终模型就是正确的。

$$
\begin{gathered}
Y_{i, t}^{D R}=g_{t}\left(X_{i}, T_{i}\right)+\frac{Y_{i}-g_{t}\left(X_{i}, T_{i}\right)}{p_{t}\left(X_{i}, T_i\right)} \\
\left\{T_{i}=t\right\}
\end{gathered}
$$

```python
def label_define(self, Y, T, X=None, W=None, *, sample_weight=None, groups=None):
    XW = self._combine(X, W)
    propensities = np.maximum(self._model_propensity.predict_proba(XW), self._min_propensity)
    n = T.shape[0]
    Y_pred = np.zeros((T.shape[0], T.shape[1] + 1))
    T_counter = np.zeros(T.shape)
    Y_pred[:, 0] = self._model_regression.predict(np.hstack([XW, T_counter])).reshape(n)
    Y_pred[:, 0] += (Y.reshape(n) - Y_pred[:, 0]) * np.all(T == 0, axis=1) / propensities[:, 0]
    for t in np.arange(T.shape[1]):
        T_counter = np.zeros(T.shape)
        T_counter[:, t] = 1
        Y_pred[:, t + 1] = self._model_regression.predict(np.hstack([XW, T_counter])).reshape(n)
        Y_pred[:, t + 1] += (Y.reshape(n) - Y_pred[:, t + 1]) * (T[:, t] == 1) / propensities[:, t + 1]
    T_complete = np.hstack(((np.all(T == 0, axis=1) * 1).reshape(-1, 1), T))
    propensities_weight = np.sum(propensities * T_complete, axis=1)
    return Y_pred.reshape(Y.shape + (T.shape[1] + 1,)), propensities_weight.reshape((n,))
```

## 参考

- [Heejung Bang and James M Robins. 2005. Doubly robust estimation in missing data and causal inference models. Biometrics. 2005.](https://www.math.mcgill.ca/dstephens/SISCR2018/Articles/bang_robins_2005.pdf)

- [Michele Jonsson Funk, Daniel Westreich, Chris Wiesen, Til Stürmer, M Alan Brookhart, and Marie Davidian. Doubly robust estimation of causal effects. American journal of epidemiology. 2011.](https://academic.oup.com/aje/article/173/7/761/103691)

- [Reddi S, Poczos B, Smola A. Doubly robust covariate shift correction. Proceedings of the AAAI Conference on Artificial Intelligence. 2015.](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9498/9965)

- [Dudik, M., Erhan, D., Langford, J., & Li, L. Doubly robust policy evaluation and optimization. Statistical Science, 2014](https://arxiv.org/pdf/1503.02834.pdf)

- https://matheusfacure.github.io/python-causality-handbook/12-Doubly-Robust-Estimation.html

- https://econml.azurewebsites.net/spec/estimation/dr.html


## 区别

Double Machine Learning (aka RLearner)
