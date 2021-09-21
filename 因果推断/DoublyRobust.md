# Doubly Robust

| 说明     | $$ DR_1 $$                                              | $$ DR_0 $$                                                      |
| -------- | ------------------------------------------------------- | --------------------------------------------------------------- |
| 通用表达 | $$ \frac{Y_{(T=1)}T}{PS} - \frac{\hat Y_1(T-PS)}{PS} $$ | $$ \frac{Y_{(T=0)}(1-T)}{1-PS} + \frac{\hat Y_0(T-PS)}{1-PS} $$ |
| T=1      | $$ \frac{Y_{(T=1)}}{PS} - \frac{\hat Y_1(1-PS)}{PS} $$  | $$ \hat Y_0 $$                                                  |
| T=0      | $$ \hat Y_1 $$                                          | $$ \frac{Y_{(T=0)}}{1-PS} - \frac{\hat Y_0 PS}{1-PS} $$         |

其中 T 表示treatment干预，PS为样本对于treatment组的倾向分，$ \hat Y_1$ 表示用户有treatment时的响应，$ \hat Y_0$ 表示用户无treatment时的响应。


## Doubly Robust Learning



## 参考

- [Heejung Bang and James M Robins. 2005. Doubly robust estimation in missing data and causal inference models. Biometrics. 2005.](https://www.math.mcgill.ca/dstephens/SISCR2018/Articles/bang_robins_2005.pdf)

- [Michele Jonsson Funk, Daniel Westreich, Chris Wiesen, Til Stürmer, M Alan Brookhart, and Marie Davidian. Doubly robust estimation of causal effects. American journal of epidemiology. 2011.](https://academic.oup.com/aje/article/173/7/761/103691)

- [Reddi S, Poczos B, Smola A. Doubly robust covariate shift correction. Proceedings of the AAAI Conference on Artificial Intelligence. 2015.](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9498/9965)

- [Dudik, M., Erhan, D., Langford, J., & Li, L. Doubly robust policy evaluation and optimization. Statistical Science, 2014](https://arxiv.org/pdf/1503.02834.pdf)


## 区别

Double Machine Learning (aka RLearner)
