## 广联达
白板日记本写代码,不能调试,人工阅卷. 4道编程题
### 广联达5.13笔试
#### 1. 数学题
模拟题,略

#### 2. 水龙头浇灌 bfs
```python
from collections import deque
def bfs(grid, k):
	n = len(grid)
	if n == 0: return None
	m = len(grid[0])
	begins = []
	for i in range(n):
		for j in range(m):
			if grid[i][j] == 1:
				begins.append((i,j))
	queue = deque(begins)
	level = -1
	searches = [(1,0),(-1,0),(0,1),(0,-1)]
	count = 0
	while queue:
		level += 1
		for i in range(len(queue)):
			row_ori, col_ori = queue.pop()
			count += 1
			for search in searches:
				row, col = row_ori+search[0], col_ori+search[1]
				if row<0 or row>= n or col<0 or col>=m:
					continue
				if grid[row][col] != 1:
					grid[row][col] = 1
					queue.appendleft((row, col))
		if level == k: break
	return m*n-count

if __name__ == "__main__":
	k = 2
	grid = [[0,1,0,0,0],
			[0,0,0,1,0],
			[0,1,0,0,0],
			[0,0,0,0,0]]
	result = bfs(grid, k)
	print(result)
```

#### 3.[437. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/)
```python
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> int:
        self.count = 0
        def helper(node, prefix, cur_sum):
            if not node: return
            cur_sum += node.val
            val = cur_sum - sum
            if val in prefix:
                self.count += prefix[val]
            prefix[cur_sum] = prefix.get(cur_sum, 0) + 1
            helper(node.left, prefix, cur_sum)
            helper(node.right, prefix, cur_sum)
            prefix[cur_sum] -= 1
        prefix = {0:1} # becareful
        helper(root, prefix, 0)
        return self.count
```

#### 补充题[560. 和为K的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/)
```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        """O(n^2)"""
        i, j = 0, 1
        n = len(nums)
        prefixsum = [0] * (n+1)
        for i in range(n):
            prefixsum[i+1] = prefixsum[i] + nums[i]
        cnt = 0
        for i in range(n+1):
            for j in range(i+1, n+1):
                if prefixsum[j] - prefixsum[i] == k:
                    cnt += 1
        return cnt
```
```python
        """O(n)前缀和 + memo, memo存储"""
        prefixsum = 0
        memo = {0:1} # becareful!
        cnt = 0
        for num in nums:
            prefixsum += num
            if prefixsum - k in memo:
                cnt += memo[prefixsum-k]
            if prefixsum in memo:
                memo[prefixsum] += 1
            else:
                memo[prefixsum] = 1
        return cnt
```
TODO: 再想想动态规划
#### 4.[312. 戳气球](https://leetcode-cn.com/problems/burst-balloons/)
```python
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        nums = [1] + nums + [1]
        n = len(nums)
        dp = [[0] * n for _ in range(n)]
        for i in range(n - 2, -1, -1):
            for j in range(i+2, n):
                max_val = 0
                for k in range(i + 1, j):
                    val = dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j]
                    max_val = max(val, max_val)
                dp[i][j] = max_val
        return dp[0][n-1]
```
