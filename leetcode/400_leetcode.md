# 400 leetcode
## First time
### Array
#### 基础题
##### [27. 移除元素](https://leetcode-cn.com/problems/remove-element/)
python pop() - O(1), pop(i) - O(n), remove(val) - O(n)
对数组进行删除增加操作用while+指针！
动态维护指针start与end，遇到=val的元素交换到尾部(题目说不用管顺序)
```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        start, end = 0, len(nums) - 1
        while start <= end:
            if nums[start] == val:
                nums[start], nums[end], end = nums[end], nums[start], end - 1
            else:
                start +=1
        return start
```
##### [26. 删除排序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)
双指针，快指针往前走，当遇到快慢指针值不一样，慢指针走一步，修改当前元素为快指针指向的元素。(注意题目限制条件，数组有序！)
```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not A:
            return 0

        newTail = 0

        for i in range(1, len(A)):
            if A[i] != A[newTail]:
                newTail += 1
                A[newTail] = A[i]

        return newTail + 1
```
##### [80. 删除排序数组中的重复项 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array-ii/)
该解法可拓展为删除有序数组的k重复项，同样可解决 leetcode 26。

如果当前元素比其第前k个元素大（如果当前元素与其第前k个元素不同），将当前元素赋值给指针停留位置，指针停留位置+1。保证nums[:i]重复不超过k个元素.
```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        i = 0
        k = 2
        for num in nums:
            if i < k or num > nums[i-k]:
                nums[i] = num
                i += 1
        return i
```
##### [189. 旋转数组](https://leetcode-cn.com/problems/rotate-array/)
方法1： O(n) in time, O(1) in space.
- reverse the first n - k elements
- reverse the rest of them
- reverse the entire array
```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if k is None or k <= 0:
            return
        k, end = k % len(nums), len(nums) - 1
        nums.reverse()  #inplace
        self.reverse(0, k-1, nums)
        self.reverse(k, end, nums) #end = len(nums)-1

    def reverse(self, start, end, nums):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start, end = start + 1, end - 1
```
方法2： O(n^2) in time, O(1) in space.
旋转k次
```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        k = k % len(nums)
        for i in range(0, k):
            tmp = nums[-1]
            for j in range(0, len(nums) - 1):
                nums[len(nums) - 1 - j] = nums[len(nums) - 2 - j]
            nums[0] = tmp
```
方法3：O(n) in time, O(1) in space.
put the last k elements in correct position (ahead) and do the remaining n - k. Once finish swap, the n and k decrease.
```python
class Solution(object):
    def rotate(self, nums, k):
        n, k, j = len(nums), k % len(nums), 0
        while n > 0 and k % n != 0:
            for i in range(0, k):
                nums[j + i], nums[len(nums) - k + i] = nums[len(nums) - k + i], nums[j + i] # swap
            n, j = n - k, j + k
            k = k % n
```
方法4: O(n) in time, O(1) in space.
```python
class Solution:
    def rotate(self, nums, k):
        n = len(nums)
        k = k % n
        nums[:] = nums[n-k:] + nums[:n-k]
```

##### [41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/)
需要再理解一下
```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        """
        after removing all the numbers greater than or equal to n,
        all the numbers remaining are smaller than n. If any number i appears,
        we add n to nums[i] which makes nums[i]>=n. Therefore, if nums[i]<n,
        it means i never appears in the array and we should return i.
        """
        nums.append(0)
        n = len(nums)
        for i in range(len(nums)): # delete those useless elements
            if nums[i]<0 or nums[i]>=n:
                nums[i]=0
        print(nums)
        for i in range(len(nums)): # use the index as the hash to record the frequency of each number
            nums[nums[i]%n]+=n
        print(nums)
        for i in range(1,len(nums)):
            if nums[i]//n==0:
                return i
        return n
```
##### [299. 猜数字游戏](https://leetcode-cn.com/problems/bulls-and-cows/)
数据结构 Counter &, |, (a&b).values()
```python
class Solution:
    def getHint(self, secret: str, guess: str) -> str:
        """ Counter &, |, (a&b).values() """
        from collections import Counter
        s, g = Counter(secret), Counter(guess)
        a = sum(i == j for i, j in zip(secret, guess))
        return '%sA%sB' % (a, sum((s & g).values()) - a)
```
```python
class Solution:
    def getHint(self, secret: str, guess: str) -> str:
        from collections import defaultdict
        secret_dict = defaultdict(list)
        guess_dict = defaultdict(list)
        for i in range(len(secret)):
            secret_dict[secret[i]].append(i)
            guess_dict[guess[i]].append(i)
        A, B = 0, 0
        for key in guess_dict:
            if key in secret_dict:
                secret_indexs = secret_dict[key]
                guess_indexs = guess_dict[key]
                if len(secret_indexs) < len(guess_indexs):
                    short, long = secret_indexs, guess_indexs
                else: short, long = guess_indexs, secret_indexs
                for i in short:
                    if i in long: A += 1
                    else: B += 1

        result = str(A)+'A'+str(B)+'B'
        return result
```

##### [134. 加油站](https://leetcode-cn.com/problems/gas-station/)
1. if sum of gas is more than sum of cost, then there must be a solution.
2. The tank should never be negative, so restart whenever there is a negative number.
```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        n = len(gas)

        total_tank, curr_tank = 0, 0
        starting_station = 0
        for i in range(n):
            total_tank += gas[i] - cost[i]
            curr_tank += gas[i] - cost[i]
            # If one couldn't get here,
            if curr_tank < 0:
                # Pick up the next station as the starting one.
                starting_station = i + 1
                # Start with an empty tank.
                curr_tank = 0

        return starting_station if total_tank >= 0 else -1
```

##### [118. 杨辉三角](https://leetcode-cn.com/problems/pascals-triangle/)
```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        triangle = []

        for row_num in range(num_rows):
            # The first and last row elements are always 1.
            row = [None for _ in range(row_num+1)]
            row[0], row[-1] = 1, 1

            # Each triangle element is equal to the sum of the elements
            # above-and-to-the-left and above-and-to-the-right.
            for j in range(1, len(row)-1):
                row[j] = triangle[row_num-1][j-1] + triangle[row_num-1][j]

            triangle.append(row)

        return triangle
```

##### [119. 杨辉三角 II](https://leetcode-cn.com/problems/pascals-triangle-ii/)
```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        """
        假设j - 1行为[1,3,3,1], 那么我们前面插入一个0(j行的数据会比j-1行多一个),
        然后执行相加[0+1,1+3,3+3,3+1,1] = [1,4,6,4,1], 最后一个1保留即可.
        """
        r = [1]
        for i in range(1, rowIndex + 1):
            r.insert(0, 0)
            for j in range(i):
                r[j] = r[j] + r[j + 1]
        return r
```

##### [169. 多数元素](https://leetcode-cn.com/problems/majority-element/)
哈希表：O(n), O(n)
```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        dict_ = {}
        half = len(nums) // 2
        for item in nums:
            if item not in dict_:
                dict_[item] = 1
            else:
                dict_[item] += 1
            if dict_[item] > half: return item
        return False
```
Boyer-Moore 投票：O(n), O(1). 数数相抵消，剩下的是众数
```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        count = 0
        candidate = None

        for num in nums:
            if count == 0:
                candidate = num
            count += (1 if num == candidate else -1)

        return candidate
```
排序: O(nlogn), O(1)
单调数组，数组长度偶数，众数坐标 n//2+1, 奇数 n//2
```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        nums = sorted(nums)
        return nums[len(nums)//2+1] if len(nums)%2 == 0 else nums[len(nums)//2]
```

##### [229. 求众数 II](https://leetcode-cn.com/problems/majority-element-ii/)
```python
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        if not nums: return []
        count1, count2, candidate1, candidate2 = 0, 0, 0, 1
        for n in nums:
            if n == candidate1:
                count1 += 1
            elif n == candidate2:
                count2 += 1
            elif count1 == 0:
                candidate1, count1 = n, 1
            elif count2 == 0:
                candidate2, count2 = n, 1
            else:
                count1, count2 = count1 - 1, count2 - 1
        return [n for n in (candidate1, candidate2) if nums.count(n) > len(nums) // 3] # 注意最后有一个对c1,c2的筛选
```

##### [274. H指数](https://leetcode-cn.com/problems/h-index)
![](assets/400_leetcode-bec248b5.png)
两种方法：1. sort，取直方图下最大正方形 2. cut为正方形，计数排序
https://leetcode-cn.com/problems/h-index/solution/hzhi-shu-by-leetcode/

##### [275. H指数 II](https://leetcode-cn.com/problems/h-index-ii)
线性
```python
class Solution:
    def hIndex(self, citations):
        n = len(citations)
        for idx, c in enumerate(citations):
            if c >= n - idx:
                return n - idx
        return 0

```
数组有序，用二分查找 时间复杂度 O(logn)
```python
class Solution:
    def hIndex(self, citations):
        n = len(citations)
        left, right = 0, n - 1
        while left <= right:
            pivot = left + (right - left) // 2
            if citations[pivot] == n - pivot:
                return n - pivot
            elif citations[pivot] < n - pivot:
                left = pivot + 1
            else:
                right = pivot - 1
        return n - left
```
https://leetcode-cn.com/problems/h-index-ii/solution/hzhi-shu-ii-by-leetcode/

##### [55. 跳跃游戏](https://leetcode-cn.com/problems/jump-game/)
https://leetcode-cn.com/problems/jump-game/solution/tiao-yue-you-xi-by-leetcode/
- 递归回溯 O(2^n)
- 递归记忆表
- 动态规划
- 贪心 （这题可以用贪心，只需要一种可能的方案即可）

##### [45. 跳跃游戏 II](https://leetcode-cn.com/problems/jump-game-ii/)

##### [121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)
只能交易一次
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) == 0: return 0
        profit_0 = 0
        profit_1 = -max(prices)
        for price in prices:
            profit_0 = max(profit_0, profit_1 + price) # 最高差值
            profit_1 = max(profit_1, - price) # 最低买入价
            print('profit_0 {}'.format(profit_0))
            print('profit_1 {}'.format(profit_1))
        return profit_0
```

##### [122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)
可以尽可能多的交易
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) == 0: return 0
        profit_0 = 0
        profit_1 = -max(prices)

        for price in prices:
            profit_0 = max(profit_0, profit_1 + price)
            profit_1 = max(profit_1, profit_0 - price)
            print('profit_0 {}'.format(profit_0))
            print('profit_1 {}'.format(profit_1))

        return profit_0
```

##### [123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)
最多可以交易两次
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) == 0: return 0
        profit_10 = 0
        profit_11 = -max(prices)
        profit_20 = 0
        profit_21 = -max(prices)

        for price in prices:
            profit_10 = max(profit_10, profit_11 + price)
            profit_11 = max(profit_11, -price)
            profit_20 = max(profit_20, profit_21 + price)
            profit_21 = max(profit_21, profit_10 - price)

        return profit_20
```

##### [188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)
```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if len(prices) == 0: return 0
        # k 太大导致爆栈, 因此如果k大于prices, 当成无限次交易处理
        if len(prices) < k:
            profit_0 = 0
            profit_1 = -max(prices)
            for price in prices:
                profit_0 = max(profit_0, profit_1 + price)
                profit_1 = max(profit_1, profit_0 - price)
            return profit_0

        profit_0 = [0 for i in range(k+1)]
        profit_1 = [0] + [-max(prices) for i in range(k+1)]

        for price in prices:
            for i in range(1, k+1):
                profit_0[i] = max(profit_0[i], profit_1[i] + price)
                profit_1[i] = max(profit_1[i], profit_0[i-1] - price)

        return profit_0[-1]
```

##### [309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) == 0: return 0
        profit_0_pre = 0
        profit_0 = 0
        profit_1 = -max(prices)

        for price in prices:
            profit_0_old = profit_0
            profit_0 = max(profit_0, profit_1 + price)
            profit_1 = max(profit_1, profit_0_pre - price)
            profit_0_pre = profit_0_old

        return profit_0
```

##### [11. 盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water)
首尾双指针，哪边低，哪边指针向内移动
```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        p0 = 0
        p1 = len(height) - 1
        if p0 == p1: return 0
        max_area = 0

        while (p0 != p1):
            area = (p1 - p0) * min(height[p0], height[p1])
            print(area)
            max_area = max(area, max_area)
            if height[p0] < height[p1]:
                p0 += 1
            else: p1 -= 1

        return max_area
```

##### [42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)
四种解法
https://leetcode-cn.com/problems/trapping-rain-water/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by-w-8/

##### [334. 递增的三元子序列](https://leetcode-cn.com/problems/increasing-triplet-subsequence/)
```python
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        first = second = float('inf')
        for n in nums:
            # 记录最小的数
            if n <= first:
                first = n
            # 记录第二小的数
            elif n <= second:
                second = n
            else:
                return True
        return False
```

#### [128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/solution/zui-chang-lian-xu-xu-lie-by-leetcode/)
1. 暴力法，遍历每个数字查询。总体复杂度O(n^3), 注意数组的 in 查询操作, 时间复杂度O(n)
```python
class Solution:
    def longestConsecutive(self, nums):
        longest_streak = 0
        for num in nums:
            current_num = num
            current_streak = 1
            while current_num + 1 in nums:
                current_num += 1
                current_streak += 1
            longest_streak = max(longest_streak, current_streak)

        return longest_streak
```

2. 先排序，再依次通过索引向后查询即可。O(nlogn)
```python
class Solution:
    def longestConsecutive(self, nums):
        if not nums:
            return 0
        nums.sort()
        longest_streak = 1
        current_streak = 1
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1]:
                if nums[i] == nums[i-1]+1:
                    current_streak += 1
                else:
                    longest_streak = max(longest_streak, current_streak)
                    current_streak = 1
        return max(longest_streak, current_streak)
```

3. 基于1利用hashmap查找，添加O(1)的特性.先构造hashmap set(). 然后如果 num - 1 not in num_set，开始查找统计。O(n). 尽管在 for 循环中嵌套了一个 while 循环，时间复杂度看起来像是二次方级别的。但其实它是线性的算法。因为只有当 currentNum 遇到了一个序列的开始， while 循环才会被执行, while 循环在整个运行过程中只会被迭代 n 次。这意味着尽管看起来时间复杂度为O(n^2) ，实际这个嵌套循环只会运行 O(n + n)次。所有的计算都是线性时间的，所以总的时间复杂度是 O(n)。
```python
class Solution:
    def longestConsecutive(self, nums):
        longest_streak = 0
        num_set = set(nums)
        for num in num_set:
            if num - 1 not in num_set:
                current_num = num
                current_streak = 1
                while current_num + 1 in num_set:
                    current_num += 1
                    current_streak += 1
                longest_streak = max(longest_streak, current_streak)
        return longest_streak
```

#### [164. 最大间距](https://leetcode-cn.com/problems/maximum-gap/)
#### [287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/)
两题排序题

#### [330. 按要求补齐数组](https://leetcode-cn.com/problems/patching-array/)
贪心，题解很巧妙
https://leetcode-cn.com/problems/patching-array/solution/an-yao-qiu-bu-qi-shu-zu-by-leetcode/

#### [4. 寻找两个有序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)
TODO: check
```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # 为了让搜索范围更小，我们始终让 num1 是那个更短的数组，PPT 第 9 张
        if len(nums1) > len(nums2):
            # 这里使用了 pythonic 的写法，即只有在 Python，中可以这样写
            # 在一般的编程语言中，得使用一个额外变量，通过"循环赋值"的方式完成两个变量的地址的交换
            nums1, nums2 = nums2, nums1

        # 上述交换保证了 m <= n，在更短的区间 [0, m] 中搜索，会更快一些
        m = len(nums1)
        n = len(nums2)

        # 使用二分查找算法在数组 nums1 中搜索一个索引 i，PPT 第 9 张
        left = 0
        right = m

        while left <= right:
            i = left + (right-left) // 2
            j = (m+n+1) // 2 - i

            # 边界值的特殊取法的原因在 PPT 第 10 张
            nums1_left_max = float('-inf') if i == 0 else nums1[i - 1]
            nums1_right_min = float('inf') if i == m else nums1[i]

            nums2_left_max = float('-inf') if j == 0 else nums2[j - 1]
            nums2_right_min = float('inf') if j == n else nums2[j]

            # 交叉小于等于关系成立，那么中位数就可以从"边界线"两边的数得到，原因在 PPT 第 2 张、第 3 张
            if nums1_left_max <= nums2_right_min and nums2_left_max <= nums1_right_min:
                # 已经找到解了，分数组之和是奇数还是偶数得到不同的结果，原因在 PPT 第 2 张
                if (m + n) % 2 == 1:
                    return max(nums1_left_max, nums2_left_max)
                else:
                    return (max(nums1_left_max, nums2_left_max) + min(nums1_right_min, nums2_right_min)) / 2
            elif nums1_left_max > nums2_right_min:
                # 这个分支缩短边界的原因在 PPT 第 8 张，情况 ②
                right = i - 1
            else:
                # 这个分支缩短边界的原因在 PPT 第 8 张，情况 ①
                left = i + 1
        raise ValueError('传入无效的参数，输入的数组不是有序数组，算法失效')
```

#### [321. 拼接最大数](https://leetcode-cn.com/problems/create-maximum-number/)
TODO: check
```python
class Solution:
    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        # 求出单个数组可以组成i位的最大数组
        def getMaXArr(nums, i):
            if not i:
                return []
            # pop表示最多可以不要nums里几个数字，要不组成不了i位数字
            stack, pop = [], len(nums) - i
            for num in nums:
                while pop and stack and stack[-1] < num :
                    pop -= 1
                    stack.pop()
                stack.append(num)
            return stack[:i]

        def merge(tmp1, tmp2):
            return [max(tmp1, tmp2).pop(0) for _ in range(k)]

        res = [0] * k
        for i in range(k + 1):
            if i <= len(nums1) and k - i <= len(nums2):
                # 取num1的i位, num2的k - i
                tmp1 = getMaXArr(nums1, i)
                tmp2 = getMaXArr(nums2, k - i)
                # 合并
                tmp = merge(tmp1, tmp2)
                if res < tmp:
                    res = tmp
        return res
```

#### [327. 区间和的个数](https://leetcode-cn.com/problems/count-of-range-sum)
不会啊, TODO:线段树

#### [289. 生命游戏](https://leetcode-cn.com/problems/game-of-life/)
遍历标记，再遍历修改

#### [57. 插入区间](https://leetcode-cn.com/problems/insert-interval/)
```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        # init data
        new_start, new_end = newInterval
        idx, n = 0, len(intervals)
        output = []

        # add all intervals starting before newInterval
        while idx < n and new_start > intervals[idx][0]:
            output.append(intervals[idx])
            idx += 1

        # add newInterval
        # if there is no overlap, just add the interval
        if not output or output[-1][1] < new_start:
            output.append(newInterval)
        # if there is an overlap, merge with the last interval
        else:
            output[-1][1] = max(output[-1][1], new_end)

        # add next intervals, merge with newInterval if needed
        while idx < n:
            interval = intervals[idx]
            start, end = interval
            idx += 1
            # if there is no overlap, just add an interval
            if output[-1][1] < start:
                output.append(interval)
            # if there is an overlap, merge with the last interval
            else:
                output[-1][1] = max(output[-1][1], end)
        return output
```

#### [239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)
注意体会双端队列的用法，左端popleft不在窗口的元素，右端pop小于当前元素的元素
```python
from collections import deque
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # base cases
        n = len(nums)
        if n * k == 0:
            return []
        if k == 1:
            return nums

        def clean_deque(i):
            # remove indexes of elements not from sliding window
            if deq and deq[0] == i - k:
                deq.popleft()

            # remove from deq indexes of all elements
            # which are smaller than current element nums[i]
            while deq and nums[i] > nums[deq[-1]]:
                deq.pop()

        # init deque and output
        deq = deque()
        max_idx = 0
        for i in range(k):
            clean_deque(i)
            deq.append(i)
            # compute max in nums[:k]
            if nums[i] > nums[max_idx]:
                max_idx = i
        output = [nums[max_idx]]

        # build output
        for i in range(k, n):
            clean_deque(i)
            deq.append(i)
            output.append(nums[deq[0]])
        return output
```

#### [209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/)
1. 暴力两次遍历 O(n^2)
2. 转为累计和，二分搜索左端点 O(nlogn)
3. 滑动窗口  O(n)


#### [238. 除自身以外数组的乘积](https://leetcode-cn.com/problems/product-of-array-except-self/)
1. 左右乘积列表 O(n), O(n)
数组 L 和 R. 对于给定索引 i，L[i] 代表的是 i 左侧所有数字的乘积，R[i] 代表的是 i 右侧所有数字的乘积。

#### [152. 乘积最大子序列](https://leetcode-cn.com/problems/maximum-product-subarray)
1. 动态规划，同时维护 min, max. O(n)

#### [228. 汇总区间](https://leetcode-cn.com/problems/summary-ranges/)
1. 一次遍历就可以了. O(n)

#### [88. 合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/)
从后往前遍历，更利于数组的修改 O(n+m)

#### [75. 颜色分类](https://leetcode-cn.com/problems/sort-colors/)
1. 基数排序 时间复杂度为O(n+k)，空间复杂度为O(n+k)。n 是待排序数组长度, k=2-0+1=3
```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        def countingSort(array):
            min_value = min(array)
            max_value = max(array)
            bucket_len = max_value -  min_value + 1
            buckets = [0] * bucket_len
            for num in array:
                buckets[num - min_value] += 1
            array.clear() # 注意不要用 array = []
            for i in range(len(buckets)):
                while buckets[i] != 0:
                    buckets[i] -= 1
                    array.append(i + min_value)

        return countingSort(nums)
```
2. 三路快排，空间复杂度O(1)
```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """三路快排的partition的稍微改动"""
        pivot = 1
        p_l = 0
        p_r = len(nums)
        p = 0
        while (p < p_r):
            if nums[p] < pivot:
                nums[p], nums[p_l] = nums[p_l], nums[p]
                p += 1
                p_l += 1
            elif nums[p] > pivot:
                p_r -= 1
                nums[p], nums[p_r] = nums[p_r], nums[p]
            else:
                p += 1
```

#### [283. 移动零](https://leetcode-cn.com/problems/move-zeroes/)
双指针，快指针向前遍历，遇到非0将慢指针赋值，慢指针+1

#### [376. 摆动序列](https://leetcode-cn.com/problems/wiggle-subsequence/)
1. 动态规划（1维dp）
2. 动态规划（登楼梯）
3. 贪心

#### []
