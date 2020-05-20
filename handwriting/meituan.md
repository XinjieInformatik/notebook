# 美团笔试题
5道编程题,2小时.题目不难,但是特别容易粗心粗,从第二题开始,对时间空间复杂度有要求.

- 难度不在与题目,而是逻辑严谨才能全部AC.
- 赛码网错误提示有问题,自己多找找原因,认真读题.
- 美团的题目,在本地调试,再复制上去比较好
- 注意标准输入输出问题! 字符串大数不要转float(int可以),会转成e科学计数法.
- 输出round是没用的,用{:.2f}.format()
- 注意赛码网,采用最后一次提交成绩
- 不要把时间拖到最后,一开始做快点,第一题不要做太久
- 不要把读取数据放到最后一刻,格式不对坑死人,直接在IDE里用input()吧

## 5.12 笔试
### 1. 单科成绩优秀奖
```
输入: 学生人数,考试科目数,每个学生的考试成绩
5 5
28 35 38 10 19
4 76 72 38 86
96 80 81 17 10
70 64 86 85 10
1 93 19 34 41
输出:
4
```
特别小心,单科可以有多个优秀奖,一名学生获多个奖,只算一个
```python
def calculate(nums, n, m):
    nums_by_less = []
    for i in range(m):
        nums_by_less.append([])
        for j in range(n):
            nums_by_less[i].append(nums[i][j])
    person = set()
    for less in nums_by_less:
        max_score = max(less)
        for i in range(len(less)):
            if less[i] == max_score:
                person.add(i)
    print(len(person))
```

### 2. 最大汉明距离
补充leetcode异或计算汉明距离
[461. 汉明距离](https://leetcode-cn.com/problems/hamming-distance/)
```python
def hammingDistance(x: int, y: int) -> int:
    xor = x ^ y
    distance = 0
    while xor:
        distance += 1
        xor = xor & (xor-1)
    return distance
```
汉明距离,二进制不同的位的数目
```
输入:
3
1 2 3
输出: 输入数组中两两元素最大的汉明距离
2
```
```python
max_distance = 0
for i in range(n):
    for j in range(i+1, n):
        distance = hammingDistance(nums[i],nums[j])
        max_distance = max(max_distance, distance)
```

### 避难所
```python
if __name__ == "__main__":
    n, m = list(map(int, input().split()))
    # print(m)
    ops, poss = [], []
    # home = [1 for _ in range(n)]
    distory = set()

    for i in range(m):
        op, pos = list(map(int, input().split()))
        if op == 1:
            # home[pos-1] = 0
            distory.add(pos-1)
        if op == 2:
            ans = None
            if pos-1 not in distory:
                ans = pos
            if pos<n and pos not in distory:
                ans = pos+1
            if ans == None:
                ans = -1
            print(ans)
```
### 字符串
注意边界条件,注意j为index+1,注意ceil里用/
```python
import math
def calculate(s, n, k):
    cnt = 0
    for i in range(n):
        for j in range(i, n+1):
            for t in range(k, math.ceil((j-i)/2)):
                if s[i:i+t] == s[j-t:j]:
                    cnt += 1
                    print(s[i:j])
    return cnt

if __name__ == "__main__":
    s = "abcabcabc"
    n = len(s)
    k = 3
    cnt = calculate(s, n, k)
    print(cnt)
```

### 相似
判断二进制是否相似,尤其注意复杂度,剪枝条件. 略

## 5.16 笔试

### 1.搜点所在行,列,有其他数字则cnt+1
```python
def get_input():
    n = int(input())
    star = []
    for i in range(n):
        x, y = list(map(int, input().split()))
        star.append((x,y))
    return n, star

def calculate(n, star):
    res = 0
    for i in range(n):
        x = star[i][0]为什么
        y = star[i][1]
        up, down, left, right = False, False, False, False
        for j in range(n):
            if i == j: continue
            if star[j][0] == x:
                if star[j][1] < y:
                    down = True
                else:
                    up = True
            if star[j][1] == y:
                if star[j][0] < x:
                    left = True
                else:
                    right = True
        if up and down and left and right:
            res += 1
    return res


if __name__ == "__main__":
    """
    8
    0 0
    0 1
    0 2
    0 3
    1 1
    1 2
    -1 1
    -1 2
    """
    n, star = get_input()
    ans = calculate(n, star)
    print(ans)
```

### 2.最短循环节
数据量10^5, 不能用O(n^2)算法,用字典

```python
def get_input():
    a,b,m,x = list(map(int, input().split()))
    return a,b,m,x

def calculate(a,b,m,x):
    res = 0
    visited = dict()
    for i in range(2*m):
        x = (a*x+b) % m
        if x not in visited:
            visited[x] = i
        else:
            res = i - visited[x]
            break
    return res

if __name__ == "__main__":
    """
    2 5 8 9
    """
    a, b, m, x = get_input()
    ans = calculate(a, b, m, x)
    print(ans)
```

### 3.规划化货币
注意不用字符串不要转float!会变成e
```python

def calculate(s):
    if len(s)>0 and s[0] == "-":
        flag = True
        s = s[1:]
    else:
        flag = False
    if "." in s:
        part1, part2 = s.split(".")
    else:
        part1 = s
        part2 = "00"
    n1 = len(part1)
    add_num = n1 // 3
    add_list = [n1-(i+1)*3 for i in range(add_num) if n1-(i+1)*3 != 0]
    for item in add_list:
        part1 = part1[:item] + "," + part1[item:]
    n2 = len(part2)
    if n2 == 1: part2 += "0"
    part2 = part2[:2]
    part1 = "$" + part1
    if flag:
        part1 = "(" + part1
        part2 = part2 + ")"
    return part1+"."+part2


if __name__ == "__main__":
    num = "928229222.9292"
    ans = calculate(num)
    print(ans)
```

#### 4.数数对
答案错误,不知道为什么
```python
def calculate2(nums, n, k):
    nums.sort()
    res, mod = divmod(k, n)
    if mod == 0: res -= 1
    ans = "(" + str(nums[res]) + "," + str(nums[mod-1]) + ")"
    return ans

if __name__ == "__main__":
    nums = [1,1,2,3,4]
    nums = list(set(nums))
    k = 11
    ans = calculate2(nums, len(nums), k)
    print(ans)
```

#### 5.最优购买策略
待完善,注意输出规范,{:.2f}
```python
def get_input():
    n, k = list(map(int, input().split()))
    return

def main():
    n, k = list(map(int, input().split()))
    sum_ = 0
    mi = 1e9
    ap = []
    bp = []
    for i in range(n):
        price, cls = list(map(int, input().split()))
        if cls == 1:
            ap.append(price)
            sum_ += price
        if cls == 2:
            mi = min(mi, price)
            bp.append(price)
            sum_ += price
    ap = sorted(ap)
    count = len(ap) - 1
    res = 0
    while k > 0 and count >= 0:
        if k == 1:
            res += min(ap[0], mi) / 2
            break
        else:
            res += ap[count] / 2
        count -= 1
        k -= 1
    ans = sum_ - res
    print("{:.2f}".format(ans))


if __name__ == "__main__":
    main()
```
