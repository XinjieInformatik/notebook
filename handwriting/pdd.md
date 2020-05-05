## pdd 往年笔试题

## 拼多多春招笔试
### [974. 和可被 K 整除的子数组](https://leetcode-cn.com/problems/subarray-sums-divisible-by-k/)
```python
import functools
class Solution:
    def subarraysDivByK(self, A: List[int], K: int) -> int:
        """dfs超时"""
        # self.cnt = 0
        # def helper(index, comsum):
        #     if index>0 and comsum % K == 0:
        #         self.cnt += 1
        #     upper_bound = len(A) if index==0 else index+1
        #     upper_bound = min(upper_bound, len(A))
        #     for i in range(index, upper_bound):
        #         helper(i+1, comsum+A[i])
        # helper(0, 0)
        # return self.cnt

        """前缀和+同余定理+排列组合数"""
        prefix = [0] * (len(A)+1)
        for i in range(1, len(prefix)):
            prefix[i] = prefix[i-1] + A[i-1]
        for i in range(len(prefix)):
            prefix[i] = prefix[i] % K
        count = {}
        # 0 是对于sum(A)的情况
        for i in range(0,len(prefix)):
            key = prefix[i]
            if key not in count:
                count[key] = 1
            else:
                count[key] += 1
        cnt = 0
        for key in count:
            conbination_num = (count[key] * (count[key]-1)) // 2
            cnt += conbination_num
        return cnt
```

## 拼多多2019秋招部分编程题合集

### 选靓号

### 种树


### 两两配对差值最小

### 回合制游戏

## 拼多多2018秋招部分编程题合集

### 列表补全

### Anniversary

### 数三角形

### 最大乘积

### 小熊吃糖


## 拼多多2018校招内推编程题汇总

### 大整数相乘

### 六一儿童节

### 迷宫寻路


#### 讨论区待整理笔试题
https://www.nowcoder.com/discuss/339198?type=post&order=time&pos=&page=1&channel=

https://www.nowcoder.com/discuss/389775?type=post&order=time&pos=&page=1&channel=
