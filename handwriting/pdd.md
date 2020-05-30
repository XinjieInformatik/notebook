# 拼多多
## 5.6笔试
1. [945. 使数组唯一的最小增量](https://leetcode-cn.com/problems/minimum-increment-to-make-array-unique/)
2. [473. 火柴拼正方形](https://leetcode-cn.com/problems/matchsticks-to-square/) 数火柴,必须最优解法,时间复杂度要求高
3. 斐波那契找循环规律
4. 最大公约数gcd
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
### 接雨水II
```python
from heapq import *
class Solution:
    def trapRainWater(self, heightMap: List[List[int]]) -> int:
        """
        水从高出往低处流，某个位置储水量取决于四周最低高度，从最外层向里层包抄，用小顶堆动态找到未访问位置最小的高度
        """
        if not heightMap:return 0
        imax = float('-inf')
        ans = 0
        heap = []
        visited = set()
        row = len(heightMap)
        col = len(heightMap[0])
        # 将最外层放入小顶堆
        # 第一行和最后一行
        for j in range(col):
            # 将该位置的高度、横纵坐标插入堆
            heappush(heap, [heightMap[0][j], 0, j])
            heappush(heap, [heightMap[row - 1][j], row - 1, j])
            visited.add((0, j))
            visited.add((row - 1, j))
        # 第一列和最后一列
        for i in range(row):
            heappush(heap, [heightMap[i][0], i, 0])
            heappush(heap, [heightMap[i][col - 1], i, col - 1])
            visited.add((i, 0))
            visited.add((i, col - 1))
        while heap:
            h, i, j = heappop(heap)
            # 之前最低高度的四周已经探索过了，所以要更新为次低高度开始探索
            imax = max(imax, h)
            # 从堆顶元素出发，探索四周储水位置
            for x, y in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
                tmp_x = x + i
                tmp_y = y + j
                # 是否到达边界
                if tmp_x < 0 or tmp_y < 0 or tmp_x >= row or tmp_y >= col or (tmp_x, tmp_y) in visited:
                    continue
                visited.add((tmp_x, tmp_y))
                if heightMap[tmp_x][tmp_y] < imax:
                    ans += imax - heightMap[tmp_x][tmp_y]
                heappush(heap, [heightMap[tmp_x][tmp_y], tmp_x, tmp_y])
        return ans
```

## 拼多多2019秋招部分编程题合集

### 选靓号
TODO: do once more
```python
def get_input():
    n, k = list(map(int,input().split()))
    nums = [int(num) for num in input()]
    return n, k, nums

if __name__ == "__main__":
    n, k, nums = get_input()
    # check same number
    same_num, min_cost, min_costs = 0, float("inf"), []
    for target in range(9):
        costs = [(i, abs(nums[i]-target)) for i in range(n)]
        costs = sorted(costs, key=lambda ele: (ele[1]))
        cost = 0
        for item in costs[:k]:
            cost += item[1]
        if cost < min_cost:
            min_cost = cost
            min_costs = costs
            same_num = target

    change_num = []
    for item in min_costs[k:]:
        change_num.append((item[0],nums[item[0]]))
    value = min_costs[k-1][1]
    select = list(filter(lambda x:x[1]<=value, min_costs))
    cnt = 0
    res = [i for i in nums]
    for i, c in select:
        if c == value and nums[i] < same_num:
            res[i] = "skip"
        else:
            res[i] = same_num if cnt<k else nums[i]
            cnt += 1
    for i in range(n-1, -1, -1):
        if res[i] == "skip":
            res[i] = same_num if cnt<k else nums[i]
            cnt +=1

    print(min_cost)
    print("".join(map(str,res)))
```

### 种树
```python
def get_input():
    n = int(input())
    trees = list(map(int,input().split()))
    return n, trees, sum(trees)

import sys
sys.setrecursionlimit(100000000)
if __name__ == "__main__":
    n, trees, m = get_input()
    def helper(res):
        if len(res) == m:
            return res
        for i in range(n):
            tree = i+1
            if trees[i] > (m-len(res)+1)//2:
                return False
            if (not res or tree != res[-1]) and trees[i]>0:
                trees[i] -= 1
                ans = helper(res+[tree])
                if ans:
                    return ans
                trees[i] += 1
        return False

    ans = helper([])
    if ans:
        ans = map(str, ans)
        print(" ".join(ans))
    else:
        print("-")
```


### 两两配对差值最小
```python
def get_input():
    n = int(input())
    arr = map(int,input().split())
    return n, list(arr)

if __name__ == "__main__":
    n, arr = get_input()
    arr = sorted(arr)
    result = []
    for i in range(n//2):
        result.append(arr[i]+arr[n-i-1])
    print(max(result)-min(result))
```

### 回合制游戏
```python
def get_input():
    hp = int(input())
    normal = int(input())
    buffed = int(input())
    return hp, normal, buffed

import math
if __name__ == "__main__":
    hp, normal, buffed = get_input()
    if buffed <= normal*2:
        turns = math.ceil(hp / normal)
        print(turns)
    else:
        turns = 0
        turns += (hp//buffed) * 2
        hp = hp % buffed
        if hp == 0:
            pass
        elif hp <= normal:
            turns += 1
        else:
            turns += 2
        print(turns)
```

## 拼多多2018秋招部分编程题合集

### 列表补全

### Anniversary

### 数三角形

### 小熊吃糖


## 拼多多2018校招内推编程题汇总

### 最大乘积
```python
def get_input():
    n = int(input())
    nums = map(int, input().split())
    return n, list(nums)

def calculate(nums):
    if len(nums) == 3:
        return nums[0] * nums[1] * nums[2]
    p = [0] * 3
    n = [0] * 2
    for num in nums:
        if num > p[2]:
            p = [p[1], p[2], num]
        elif p[1] < num <= p[2]:
            p = [p[1], num, p[2]]
        elif p[0] < num <= p[1]:
            p = [num, p[1], p[2]]
        elif num <= n[0] and num < 0:
            n = [num, n[0]]
        elif n[0] < num < n[1] and num < 0:
            n = [n[0], num]
    return max(n[0] * n[1] * p[2], p[0] * p[1] * p[2])

if __name__ == "__main__":
    n, nums = get_input()
    ans = calculate(nums)
    print(ans)
```

### 六一儿童节
```python
def get_input():
    n_h = int(input())
    h = map(int, input().split())
    n_w = int(input())
    w = map(int, input().split())
    return n_w, list(w), n_h, list(h)


if __name__ == "__main__":
    n_w, w_list, n_h, h_list = get_input()
    cnt = 0
    for w in w_list:
        max_index, max_value = -1, 0
        for i in range(n_h):
            value = h_list[i]
            if value <= w and value > max_value:
                max_index = i
                max_value = value
        if max_index >= 0:
            h_list[max_index] = -1
            cnt += 1
    print(cnt)
```

### 迷宫寻路
只通过40%
```python
from collections import deque

def get_input():
    rows, cols = list(map(int, input().split()))
    grids = []
    start, end = [], []
    for i in range(rows):
        row = []
        content = input()
        for j in range(len(content)):
            s = content[j]
            row.append(s)
            if s == "2":
                start = [i, j]
            if s == "3":
                end = [i, j]
        grids.append(row)
    return grids, rows, cols, start, end

def bfs(grids, rows, cols, start, end):
    queue = deque([(start[0], start[1], 0, ())])
    visited = set()
    visited.add((start[0], start[1], ()))
    directions = [(1,0),(-1,0),(0,1),(0,-1)]
    while queue:
        for _ in range(len(queue)):
            row, col, step, keys = queue.pop()
            for direction in directions:
                next_row = row + direction[0]
                next_col = col + direction[1]
                add_keys = keys # important!
                if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                    continue
                cell = grids[next_row][next_col]
                if cell == "0":
                    continue
                if cell.islower() and cell not in add_keys:
                    add_keys = add_keys + (cell,)
                if cell.isupper() and cell.lower() not in add_keys:
                    continue
                if (next_row, next_col, add_keys) in visited:
                    continue
                if next_row == end[0] and next_col == end[1]:
                    return step+1
                queue.appendleft((next_row, next_col, step+1, add_keys))
                visited.add((next_row, next_col, add_keys))
    return -1

if __name__ == "__main__":
    grids, rows, cols, start, end = get_input()
    step = bfs(grids, rows, cols, start, end)
    print(step)
```

40%
```python
from collections import deque

mapping = { 'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g':6, 'h':7, 'i':8, 'j':9}
def get_input():
    rows, cols = list(map(int, input().split()))
    grids = []
    start, end = [], []
    ks = 0
    for i in range(rows):
        row = []
        content = input()
        for j in range(len(content)):
            s = content[j]
            row.append(s)
            if s == "2":
                start = [i, j]
            if s == "3":
                end = [i, j]
            #if s >= 'a' and ch <= 'j':
            #    ks |= (1 << mapping[ch])
        grids.append(row)
    return grids, rows, cols, start, end

def bfs(grids, rows, cols, start, end):
    queue = deque([(start[0], start[1], 0, 0)])
    visited = [[[None for _ in range(1 << 10)] for _ in range(cols)] for _ in range(rows)]
    visited[start[0]][start[1]][0] = True
    directions = [(1,0),(-1,0),(0,1),(0,-1)]
    while queue:
        for _ in range(len(queue)):
            row, col, step, keys = queue.pop()
            for direction in directions:
                add_keys = keys
                next_row = row + direction[0]
                next_col = col + direction[1]
                if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                    continue
                cell = grids[next_row][next_col]
                if cell == "0":
                    continue
                if cell.islower():
                    add_keys |= (1 << mapping[cell])
                if cell.isupper() and add_keys & (1 << mapping[cell.lower()]) == 0:
                    continue
                if visited[next_row][next_col][add_keys]:
                    continue
                if next_row == end[0] and next_col == end[1]:
                    return step+1
                queue.appendleft((next_row, next_col, step+1, add_keys))
                visited[next_row][next_col][add_keys] = True
    return -1

if __name__ == "__main__":
    grids, rows, cols, start, end = get_input()
    step = bfs(grids, rows, cols, start, end)
    print(step)
```

#### 讨论区待整理笔试题
https://www.nowcoder.com/discuss/339198?type=post&order=time&pos=&page=1&channel=

https://www.nowcoder.com/discuss/389775?type=post&order=time&pos=&page=1&channel=
