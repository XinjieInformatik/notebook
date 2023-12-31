# Attention is all you need
https://arxiv.org/abs/1706.03762

## 代码实现
- [pytorch implementation](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [tensorflow implementation](https://tensorflow.google.cn/tutorials/text/transformer)

## 理解transformer
- https://baijiahao.baidu.com/s?id=1651219987457222196&wfr=spider&for=pc
- https://zhuanlan.zhihu.com/p/59629215

### self-attention
输入：多个向量x组成的矩阵X
输出：attention进行weight后的矩阵Z
```python
# X, Q, K, V每一行表示一个embedding
Q = X @ W_q
K = X @ W_k
V = X @ W_v
Z = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
```

### multihead-attention
由多个self-attention组成，对于h个self-attention输出h个矩阵Z，concat后传入linear层，输出最终的Z(维度和最初的X矩阵一致)

norm：Layer Normalize
add：依然使用残差连接

### position embedding
```test
PE(pos, 2i) = sin(pos / (1e4^(2i/d)))
PE(pos, 2i+1) = cos(pos / (1e4^(2i/d)))
```


