# Leetcode python

## 栈
#### [20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)
符号匹配用单个栈
```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        match = {'{':'}', '[':']', '(':')'}
        for item in s:
            if item in match.keys():
                stack.append(item)
            else:
                if len(stack) != 0:
                    if match[stack[-1]] == item:
                        stack.pop()
                    else: return False
                else: return False
        if len(stack) == 0: return True
        else: return False
```

#### [394. 字符串解码](https://leetcode-cn.com/problems/decode-string/)
单个栈
```python
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        str_out = ""

        for index, item in enumerate(s):
            if item != ']':
                stack.append(item)
            else:
                str_temp = ""
                str_num = ""
                count = 1
                sign = stack[-1]

                while (sign != '['):
                    str_temp += stack.pop()
                    sign = stack[-1]

                stack.pop() # delete '['
                sign = stack[-1]

                while (sign.isdigit()):
                    str_num += stack.pop()
                    if stack:
                        sign = stack[-1]
                    else:
                        sign = "end"

                str_num = str_num[::-1]
                str_temp = str_temp[::-1]

                try:
                    num = int(str_num)
                except:
                    num = 1
                str_temp *= num
                for item in str_temp:
                    stack.append(item)

        if stack:
            str_out = ''.join(stack)

        return str_out
```

#### [739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/)
包含index的单调递减栈
```python
class Solution:
    def dailyTemperatures(self, T: List[int]) -> List[int]:
        result = [0] * len(T)
        stack = []

        for index, item in enumerate(T):
            # while 维护单调栈
            while stack and item > stack[-1][1]:
                i, value = stack.pop()
                res = index - i
                result[i] = res

            stack.append((index, item))

        return result
```
#### [496. 下一个更大元素 I](https://leetcode-cn.com/problems/next-greater-element-i/)
```python
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        memo = {}
        stack = []
        # 维护递减的单调栈
        for num in nums2:
            while stack and num > stack[-1]:
                val = stack.pop()
                memo[val] = num
            stack.append(num)

        result = []
        for num in nums1:
            if num in memo:
                result.append(memo[num])
            else:
                result.append(-1)

        return result
```

#### [503. 下一个更大元素 II](https://leetcode-cn.com/problems/next-greater-element-ii/)
```python
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [-1 for _ in range(n)]
        nums = nums * 2
        stack = []
        for i, num in enumerate(nums):
            while stack and num > stack[-1][1]:
                index, val = stack.pop()
                if index < n:
                    result[index] = num
            stack.append((i, num))
        return result
```

#### [556. 下一个更大元素 III](https://leetcode-cn.com/problems/next-greater-element-iii/)
求下一个全排列，一个全部倒序的数没有下一个全排列。
- 从后往前遍历找到第一个非逆序的数,inv_index。stack 逆序存储一个递增的单调栈
- 从第一个非逆序的数往后找到第一个大于它的数(可以用二分查找优化)
- 交换位置，第一个数往后逆序排序，因为已经使用栈，因此不用再逆序了
见官方题解动画 https://leetcode-cn.com/problems/next-greater-element-iii/solution/xia-yi-ge-geng-da-yuan-su-iii-by-leetcode/

```python
class Solution:
    def nextGreaterElement(self, n: int) -> int:
        str_n = str(n)
        len_n = len(str_n)
        stack = []
        inv_index = None
        for i in range(len_n-1, -1, -1):
            val = int(str_n[i])
            if stack and val < stack[-1]:
                inv_index = i
                stack.insert(0, val)
                break
            stack.append(val)

        if inv_index != None:
            ex_index = 0
            for i in range(1, len(stack)):
                if stack[i] > stack[0]:
                    ex_index = i
                    break
            stack[ex_index], stack[0] = stack[0], stack[ex_index]
            str_n_new = str_n[:inv_index] + "".join(map(str, stack))
            n_new = int(str_n_new)
            return n_new if n_new < 1<<31 else -1
        else:
            return -1
```

#### [31. 下一个排列](https://leetcode-cn.com/problems/next-permutation/)
与上一题唯一不同就是原地修改
```python
class Solution:
    def upper_bound(self, arr, left, right, target):
        while left < right:
            mid = left + (right-left) // 2
            if arr[mid] <= target:
                left = mid + 1
            else:
                right = mid
        return left

    def nextPermutation(self, nums: List[int]) -> None:
        inv_index = None
        for i in range(len(nums)-1,0,-1):
            if nums[i] > nums[i-1]:
                inv_index = i
                break
        if inv_index != None:
            nums[inv_index:] = sorted(nums[inv_index:])
            swap_index = self.upper_bound(nums, inv_index, len(nums), nums[inv_index-1])
            nums[inv_index-1], nums[swap_index] = nums[swap_index], nums[inv_index-1]
        else:
            nums.sort()
```

#### [42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)
超时
```python
class Solution:
    def trap(self, height: List[int]) -> int:
        from collections import deque
        if len(height) == 0: return 0
        max_h = max(height)
        water_count = 0
        for level_h in range(1,max_h+1):
            stack = deque([])
            for index in range(len(height)):
                if len(stack)==0 and height[index] >= level_h:
                    stack.append(index)
                elif len(stack)==1 and height[index] >= level_h:
                    if index-stack[-1] > 1:
                        stack.append(index)
                    else:
                        stack[0] = index
                if len(stack)==2:
                    left_index = stack.popleft()
                    right_index = stack[0]
                    water_count += (right_index - left_index - 1)
        return water_count
```
```python
class Solution:
    def trap(self, height: List[int]) -> int:
      """维护一个高度单调递减的栈"""
        if len(height) == 0: return 0
        water_count = 0
        stack = []
        for index in range(len(height)):
            cur_height = height[index]
            # 维护一个高度单调递减的栈
            while stack and cur_height > height[stack[-1]]:
                top = stack.pop()
                if len(stack)==0: break
                h = min(height[stack[-1]], cur_height) - height[top]
                dist = index - stack[-1] - 1
                water_count += dist * h
            stack.append(index)
        return water_count
```
#### [85. 最大矩形](https://leetcode-cn.com/problems/maximal-rectangle/)
```python
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        rows = len(matrix)
        if rows == 0: return 0
        cols = len(matrix[0])
        heights = [0] * (cols+2)
        max_area = 0
        for row in range(rows):
            stack = [0]
            for col in range(cols+2):
                if col < cols:
                    if matrix[row][col] == "1":
                        heights[col+1] += 1
                    else:
                        heights[col+1] = 0
                while heights[col] < heights[stack[-1]]:
                    cur_h = heights[stack.pop()]
                    cur_w = col - stack[-1] - 1
                    max_area = max(max_area, cur_h * cur_w)
                stack.append(col)
        return max_area
```

## 堆
#### [347. 前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements)
这题是对**堆，优先队列**很好的练习，因此有必要自己用python实现研究一下。**堆 处理海量数据的topK，分位数**非常合适，**优先队列**应用在元素优先级排序，比如本题的频率排序非常合适。与基于比较的排序算法 时间复杂度**O(nlogn)** 相比, 使用**堆，优先队列**复杂度可以下降到 **O(nlogk)**,在总体数据规模 n 较大，而维护规模 k 较小时，时间复杂度优化明显。

**堆，优先队列**的本质其实就是个完全二叉树，有其下重要性质
ps: 堆heap[0]插入一个占位节点,此时堆顶为index为1的位置,可以更方便的运用位操作.
[1,2,3] -> [0,1,2,3]
1. 父节点index为 i.
2. 左子节点index为 i << 1
3. 右子节点index为 i << 1 | 1
4. 大顶堆中每个父节点大于子节点，小顶堆每个父节点小于子节点
5. 优先队列以优先级为堆的排序依据
因为性质1，2，3，堆可以用数组直接来表示，不需要通过链表建树。

**堆，优先队列** 有两个重要操作，时间复杂度均是 O(logk)。以小顶锥为例：
1. 上浮sift up: 向堆尾新加入一个元素，堆规模+1，依次向上与父节点比较，如小于父节点就交换。
2. 下沉sift down: 从堆顶取出一个元素（堆规模-1，用于堆排序）或者更新堆中一个元素（本题），依次向下与子节点比较，如大于子节点就交换。

对于topk 问题：**最大堆求topk小，最小堆求topk大。**
- topk小：构建一个k个数的最大堆，当读取的数小于根节点时，替换根节点，重新塑造最大堆
- topk大：构建一个k个数的最小堆，当读取的数大于根节点时，替换根节点，重新塑造最小堆

**这一题的总体思路** 总体时间复杂度 **O(nlogk)**
- 遍历统计元素出现频率. O(n)
- 前k个数构造**规模为k+1的最小堆** minheap. O(k). 注意+1是因为占位节点.
- 遍历规模k之外的数据，大于堆顶则入堆，下沉维护规模为k的最小堆 minheap. O(nlogk)
- (如需按频率输出，对规模为k的堆进行排序)

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        def sift_down(arr, root, k):
            """下沉log(k),如果新的根节点>子节点就一直下沉"""
            val = arr[root] # 用类似插入排序的赋值交换
            while root<<1 < k:
                child = root << 1
                # 选取左右孩子中小的与父节点交换
                if child|1 < k and arr[child|1][1] < arr[child][1]:
                    child |= 1
                # 如果子节点<新节点,交换,如果已经有序break
                if arr[child][1] < val[1]:
                    arr[root] = arr[child]
                    root = child
                else:
                    break
            arr[root] = val

        def sift_up(arr, child):
            """上浮log(k),如果新加入的节点<父节点就一直上浮"""
            val = arr[child]
            while child>>1 > 0 and val[1] < arr[child>>1][1]:
                arr[child] = arr[child>>1]
                child >>= 1
            arr[child] = val

        stat = collections.Counter(nums)
        stat = list(stat.items())
        heap = [(0,0)]

        # 构建规模为k+1的堆,新元素加入堆尾,上浮
        for i in range(k):
            heap.append(stat[i])
            sift_up(heap, len(heap)-1)
        # 维护规模为k+1的堆,如果新元素大于堆顶,入堆,并下沉
        for i in range(k, len(stat)):
            if stat[i][1] > heap[1][1]:
                heap[1] = stat[i]
                sift_down(heap, 1, k+1)
        return [item[0] for item in heap[1:]]
```

```python
heapq 构造小顶堆, 若从大到小输出, heappush(-val)
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        import heapq
        from collections import Counter

        freq = Counter(nums)
        heap = []
        for key, val in freq.items():
            heapq.heappush(heap, (-val, key))
        result = []
        for _ in range(k):
            result.append(heapq.heappop(heap)[1])

        return result
```

再附上堆排序(从小到大输出),注意这里是大顶堆
1. 从后往前非叶子节点下沉，依次向上保证每一个子树都是大顶堆,构造大顶锥
2. 依次把大顶堆根节点与尾部节点交换(不再维护,堆规模-1),新根节点下沉。

```python
def heapSort(arr):
    def sift_down(arr, root, k):
        val = arr[root]
        while root<<1 < k:
            chlid = root << 1
            if chlid|1 < k and arr[chlid|1] > arr[chlid]:
                chlid |= 1
            if arr[chlid] > val:
                arr[root] = arr[chlid]
                root = chlid
            else:
                break
        arr[root] = val

    arr = [0] + arr
    k = len(arr)
    for i in range((k-1)>>1, 0, -1):
        sift_down(arr, i, k)
    for i in range(k-1, 0, -1):
        arr[1], arr[i] = arr[i], arr[1]
        sift_down(arr, 1, i)
    return arr[1:]
```

更多的几个堆的练习
[295. 数据流的中位数](https://leetcode-cn.com/problems/find-median-from-data-stream/)
[215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)
[面试题40. 最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)
[347. 前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements)

#### [1353. 最多可以参加的会议数目](https://leetcode-cn.com/problems/maximum-number-of-events-that-can-be-attended/)
```python
class Solution:
    def maxEvents(self, events: List[List[int]]) -> int:
        ans = 0
        end = []
        # 按起始日期从大到小排序
        events = sorted(events,reverse=True)
        for i in range(1,100001,1):
            # 如果当前日期==会议起始日期，将结束日期加入小顶堆
            while events and events[-1][0] == i:
                heapq.heappush(end, events.pop()[1])
            # 将堆中所有结束日期小于当前日期的会议pop
            while end and end[0] < i:
                heapq.heappop(end)
            # 如果堆非空，当前日期参加结束日期最小的
            if end:
                heapq.heappop(end)
                ans += 1
        return ans
```

#### [面试题40. 最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)
```python
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        def sift_up(arr, k):
            new_index, new_val = k-1, arr[k-1]
            while (new_index>0 and arr[(new_index-1)//2]<new_val):
                arr[new_index] = arr[(new_index-1)//2]
                new_index = (new_index-1)//2
            arr[new_index] = new_val

        def sift_down(arr, root, k):
            root_val = arr[root]
            while (2*root+1 < k):
                child = 2*root+1
                if child+1 < k and arr[child] < arr[child+1]:
                    child += 1
                if root_val < arr[child]:
                    arr[root] = arr[child]
                    root = child
                else: break
            arr[root] = root_val

        if k == 0: return []
        max_heap = []
        for i in range(k):
            max_heap.append(arr[i])
            sift_up(max_heap, i+1)

        for item in arr[k:]:
            if item < max_heap[0]:
                max_heap[0] = item
                sift_down(max_heap, 0, k)

        return max_heap
```

#### [295. 数据流的中位数](https://leetcode-cn.com/problems/find-median-from-data-stream/)
```python
class MaxHeap:
    def __init__(self):
        self.heap = []

    def sift_down(self, root, k):
        root_val = self.heap[root]
        while (2*root+1 < k):
            child = 2 * root + 1
            if child+1 < k and self.heap[child] < self.heap[child+1]:
                child += 1
            if root_val < self.heap[child]:
                self.heap[root] = self.heap[child]
                root = child
            else: break
        self.heap[root] = root_val

    def sift_up(self, k):
        new_index, new_val = k-1, self.heap[k-1]
        while (new_index > 0 and self.heap[(new_index-1)//2] < new_val):
            self.heap[new_index] = self.heap[(new_index-1)//2]
            new_index = (new_index-1)//2
        self.heap[new_index] = new_val

    def add_new(self, new_val):
        self.heap.append(new_val)
        self.sift_up(len(self.heap))

    def take(self):
        val = self.heap[0]
        self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
        self.heap.pop()
        if self.heap:
            self.sift_down(0, len(self.heap))
        return val

    def __len__(self):
        return len(self.heap)


class MinHeap(MaxHeap):
    def __init__(self):
        self.heap = []

    def sift_down(self, root, k):
        root_val = self.heap[root]
        while (2*root+1 < k):
            child = 2 * root + 1
            if child+1 < k and self.heap[child] > self.heap[child+1]:
                child += 1
            if root_val > self.heap[child]:
                self.heap[root] = self.heap[child]
                root = child
            else: break
        self.heap[root] = root_val

    def sift_up(self, k):
        new_index, new_val = k-1, self.heap[k-1]
        while (new_index > 0 and self.heap[(new_index-1)//2] > new_val):
            self.heap[new_index] = self.heap[(new_index-1)//2]
            new_index = (new_index-1)//2
        self.heap[new_index] = new_val


class MedianFinder:
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.max_heap = MaxHeap()
        self.min_heap = MinHeap()
        self.max_capacity =  4

    def addNum(self, num: int) -> None:
        self.max_heap.add_new(num)
        self.min_heap.add_new(self.max_heap.take())
        if len(self.max_heap) < len(self.min_heap):
            self.max_heap.add_new(self.min_heap.take())


    def findMedian(self) -> float:
        median = self.max_heap.heap[0] if len(self.max_heap) > len(self.min_heap) else (self.max_heap.heap[0]+self.min_heap.heap[0])/2
        return median

# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
```

#### [215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)
1. 规模为k的最小堆
```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def sift_down(arr, root, k):
            """ root i, l 2i, r 2i+1 """
            val = arr[root]
            while root << 1 < k:
                child = root << 1
                if child|1 < k and arr[child|1] < arr[child]:
                    child |= 1
                if arr[child] < val:
                    arr[root] = arr[child]
                    root = child
                else:
                    break
            arr[root] = val

        def sift_up(arr, k):
            child, val = k-1, arr[k-1]
            while child > 1 and arr[child>>1] > val:
                root = child >> 1
                arr[child] = arr[root]
                child = root
            arr[child] = val

        heap = [0]
        for i in range(k):
            heap.append(nums[i])
            sift_up(heap, len(heap))
        for i in range(k, len(nums)):
            if nums[i] > heap[1]:
                heap[1] = nums[i]
                sift_down(heap, 1, len(heap))
        return heap[1]
```
```python
        import heapq
        heap = []
        for i in range(k):
            heapq.heappush(heap, nums[i])
        n = len(nums)
        for i in range(k, n):
            if nums[i] > heap[0]:
                heapq.heappop(heap)
                heapq.heappush(heap, nums[i])
        return heap[0]
```
2. partition 直到 pivot_index = n-k, 可保证左边均小于pivot, 右边均大于等于pivot
快速选择可以用于查找中位数，任意第k大的数
在输出的数组中，pivot_index达到其合适位置。所有小于pivot_index的元素都在其左侧，所有大于或等于的元素都在其右侧。如果是快速排序算法，会在这里递归地对两部分进行快速排序，时间复杂度为 O(NlogN)。快速选择由于知道要找的第 N - k 小的元素在哪部分中，不需要对两部分都做处理，这样就将平均时间复杂度下降到 O(N)。
3. 注意输入的nums数组是被修改过的
```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        import random
        def qselect(arr, l, r, k_smallest):
            def partition(arr, l, r):
                i = random.randint(l, r-1)
                arr[l], arr[i] = arr[i], arr[l]
                pivot, val = l, arr[l]
                for i in range(l+1, r):
                    if arr[i] < val:
                        pivot += 1
                        arr[i], arr[pivot] = arr[pivot], arr[i]
                arr[l], arr[pivot] = arr[pivot], arr[l]
                return pivot
            def partition2(arr, left, right):
                i = random.randint(left, right-1)
                arr[left], arr[i] = arr[i], arr[left]
                val = arr[left]
                l = left + 1
                r = right - 1
                while l <= r:
                    while l < right and arr[l] <= val:
                        l += 1
                    while r > left and arr[r] >= val:
                        r -= 1
                    if l < r:
                        arr[l], arr[r] = arr[r], arr[l]
                arr[left], arr[r] = arr[r], arr[left]
                return r

            while l < r:
                pivot = partition(arr, l, r)
                if pivot < k_smallest:
                    l = pivot + 1
                else:
                    r = pivot
            return l
        n = len(nums)
        index = qselect(nums, 0, n, n-k)
        return nums[index]
```


## 队列
### 双向队列
#### [239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)
> TODO: 用动态规划再做一次

双向队列
![](assets/leetcode-632a6930.png)
```python
from collections import deque

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # 建立双向队列,储存索引便于滑窗判断
        window = deque(); res = []
        for i in range(len(nums)):
            # 构建单调队列的push操作，注意用nums恢复索引
            while window and nums[window[-1]] <= nums[i]:
                window.pop()
            window.append(i)
            # 从k-1开始res append()
            if i >= k - 1:
                res.append(nums[window[0]])
            # 如果单调队列最大值落于滑窗之外，popleft()
            if window[0] == i-k+1:
                window.popleft()
        return res
```

## 动态规划
用额外的空间，存储子问题的最优解，找到状态转移方程，不断推出当前最优解。
1. 状态转移方程
2. 初始值
#### [55. 跳跃游戏](https://leetcode-cn.com/problems/jump-game)
动态规划, 维护max_step作为能够最远跳达的位置,如果当前index<=max_step, 用nums[i]+i更新max_step
```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        max_step = 0
        n = len(nums)
        for i in range(n):
            if i <= max_step:
                nxt = nums[i]+i
                max_step = max(max_step, nxt)
                if max_step >= n-1:
                    return True
        return False
```

#### [45. 跳跃游戏 II](https://leetcode-cn.com/problems/jump-game-ii/)

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        max_step, cnt, last = 0, 0, 0
        n = len(nums)
        # 注意是n-1,检查到倒数第二个节点即可
        for i in range(n-1):
            max_step = max(max_step, i+nums[i])
            if i == last:
                cnt += 1
                last = max_step
        return cnt

        # import functools
        # n = len(nums)
        # @functools.lru_cache(None)
        # def helper(index):
        #     if index >= n-1:
        #         return 0
        #     min_step = float("inf")
        #     for i in range(nums[index], 0, -1):
        #         step = helper(index+i) + 1
        #         min_step = min(min_step, step)
        #     return min_step
        # return helper(0)
```

#### [70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)
动态规划, 它的最优解可以从其子问题的最优解来有效地构建。

第 `i` 阶可以由以下两种方法得到：

- 在第 `(i-1)` 阶后向上爬 1 阶。

- 在第 `(i-2)` 阶后向上爬 2 阶。

所以到达第 `i` 阶的方法总数就是到第 `i-1` 阶和第 `i-2` 阶的方法数之和。

令 `dp[i]` 表示能到达第 `i` 阶的方法总数，
状态转移方程（同斐波那契数）：
`dp[i] = dp[i−1] + dp[i−2]`
```python
class Solution:
    def climbStairs(self, n: int) -> int:
        f0 = 1
        f1 = 2
        if n == 1: return f0
        if n == 2: return f1
        for i in range(n-2):
            f2 = f0 + f1
            f0 = f1
            f1 = f2
        return f2
```

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        """ functools.lru_cache 用于回溯时，
        将已访问节点的值放入memo避免重复计算,
        重复节点不会再访问"""
        import functools
        @functools.lru_cache(None)
        def helper(step):
            print(step)
            if step == 0:
                return 1
            if step < 0:
                return 0
            res = 0
            for i in range(1,3):
                res += helper(step-i)
            return res

        return helper(n)
```

#### [198. 打家劫舍](https://leetcode-cn.com/problems/house-robber)
状态转移方程 cur_max = max(pprev_max + nums[i], prev_max)
```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        # 动态规划两部曲，1.定义初始值 2.定义状态转移方程
        cur_max = 0
        prev_max = 0
        pprev_max = 0

        for i in range(len(nums)):
            cur_max = max(pprev_max + nums[i], prev_max)
            pprev_max = prev_max
            prev_max = cur_max

        return cur_max

class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0: return 0
        if n < 3: return max(nums)
        dp = [0] * (n+1)
        dp[1] = nums[0]
        for i in range(2, n+1):
            steal_pre = dp[i-1]
            steal_this = dp[i-2] + nums[i-1]
            dp[i] = max(steal_pre, steal_this)

        return dp[-1]
```

#### [213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/)
```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0: return 0
        if n < 3: return max(nums)

        def helper(amounts):
            n = len(amounts)
            if n == 1: return amounts[0]
            dp = [0] * (n+1)
            dp[1] = amounts[0] # becareful
            for i in range(2, n+1):
                steal_pre = dp[i-1]
                steal_this = dp[i-2] + amounts[i-1]
                dp[i] = max(steal_pre, steal_this)
            return dp[-1]

        return max(helper(nums[1:]), helper(nums[:-1]))
```

#### [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        prev = 0
        max_sum = min(nums)
        for i in range(n):
            cur = max(nums[i], prev+nums[i])
            max_sum = max(max_sum, cur)
            prev = cur
        return max_sum
```

#### [300. 最长上升子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)
```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        """O(n^2)双重循环遍历,遍历中更新povit,维护povit_prev"""
        n = len(nums)
        max_len = 0
        for i in range(n):
            cnt = 1
            povit = povit_prev = nums[i]
            for j in range(i+1, n):
                if povit_prev < nums[j] < povit:
                    povit = nums[j]
                elif nums[j] > povit:
                    povit_prev = povit
                    povit = nums[j]
                    cnt += 1
            max_len = max(max_len, cnt)
            if max_len >= (n-i): break
        return max_len
```
最长上升子序列决定了可以在子序列里二分查找！
```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        """O(nlogn). 用dp存储遍历到i的最长上升子序列，并二分查找更新dp,
        如果index大于lenth,拓展lenth"""
        def low_bound(arr, right, target):
            left, right = 0, right
            while left < right:
                mid = left + (right-left)//2
                if arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid
            return left

        dp = [0] * len(nums)
        lenth = 0
        for num in nums:
            index = low_bound(dp, lenth, num)
            if index < lenth:
                dp[index] = num
            else:
                dp[index] = num
                lenth += 1
        return lenth
```

#### [152. 乘积最大子序列](https://leetcode-cn.com/problems/maximum-product-subarray/)
```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        curr_min = 1
        curr_max = 1
        max_value = max(nums)
        for item in nums:
            # 如果遇到负数，最大变最小，最小变最大
            if item < 0:
                curr_min, curr_max = curr_max, curr_min
            curr_max = max(item, curr_max*item)  # 无负数阶段的当前最大值
            curr_min = min(item, curr_min*item)  # 维护连乘最小值或者当前值
            max_value = max(curr_max, max_value)
        return max_value
```


#### [64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum)
逆序，二维
```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        grid_h = len(grid)
        grid_w = len(grid[0])
        dp = [[0] * grid_w for i in range(grid_h)]

        for i in range(grid_h-1, -1, -1):
            for j in range(grid_w-1, -1, -1):
                if i+1 > grid_h-1 and j+1 > grid_w-1:
                    dp[i][j] = grid[i][j]
                elif i+1 > grid_h-1 and j+1 <= grid_w-1:
                    dp[i][j] = grid[i][j] + dp[i][j+1]
                elif i+1 <= grid_h-1 and j+1 > grid_w-1:
                    dp[i][j] = grid[i][j] + dp[i+1][j]
                else:
                    dp[i][j] = grid[i][j] + min(dp[i+1][j], dp[i][j+1])

        return dp[0][0]
```

#### [62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)
```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0]*m for i in range(n)]
        for i in range(m):
            for j in range(n):
                if j-1 < 0 and i - 1 < 0:
                    dp[j][i] = 1
                elif j-1 < 0:
                    dp[j][i] = dp[j][i-1]
                elif i-1 < 0:
                    dp[j][i] = dp[j-1][i]
                else:
                    dp[j][i] = dp[j-1][i] + dp[j][i-1]
        return dp[n-1][m-1]
```

#### [221. 最大正方形](https://leetcode-cn.com/problems/maximal-square/)
用二位dp存储最大边长信息，当前网格的值为min(dp[左上，左，上])，遍历记录最大边长即可
注意对越界的限制
$$
\mathrm{dp}(i, j)=\min (\mathrm{dp}(i-1, j), \mathrm{dp}(i-1, j-1), \mathrm{dp}(i, j-1))+1
$$

![](assets/markdown-img-paste-20200103141305216.png)

```python
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        rows = len(matrix)
        if not rows: return 0
        cols = len(matrix[0])
        dp = [[0]*cols for i in range(rows)]
        max_area = 0
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == "1":
                    topleft = dp[i-1][j-1] if i-1>=0 and j-1>=0 else 0
                    top = dp[i-1][j] if i-1>=0 else 0
                    left = dp[i][j-1] if j-1>=0 else 0
                    dp[i][j] = min(topleft, top, left) + 1
                    max_area = max(max_area, dp[i][j])
        # print(dp)
        return max_area**2
```

#### [139. 单词拆分](https://leetcode-cn.com/problems/word-break/)
对于给定的字符串（s）可以被拆分成子问题 s1 和 s2, 如果这些子问题都可以独立地被拆分成符合要求的子问题，那么整个问题 s 也可以满足.
```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        dp = [1] + [0] * len(s)
        for i in range(1, len(s)+1):
            for j in range(i):
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True
        return dp[len(s)]

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        dp = [0]
        for i in range(len(s)):
            for index in dp:
                if s[index:i+1] in wordDict and (i+1) not in dp:
                    dp.append(i+1)
        if dp[-1] == len(s): return True
        else: return False
```
```python
import functools
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        lenths = [len(item) for item in wordDict]
        lenths = set(lenths)
        wordDict = set(wordDict)
        n = len(s)
        @functools.lru_cache(None)
        def helper(index):
            if index == n:
                return True
            for lenth in lenths:
                if index+lenth <= n and s[index:index+lenth] in wordDict:
                    if helper(index+lenth):
                        return True
            return False
        return helper(0)
```


#### [96. 不同的二叉搜索树](https://leetcode-cn.com/problems/unique-binary-search-trees)
G(n): 长度为n的序列的不同二叉搜索树个数

F(i,n): 以i为根的不同二叉搜索树的个数(1<=i<=n)

$$ G(n) = \sum_{i=1}^n F(i,n) $$
$$ F(i,n) = G(i-1) G(n-i) $$
$$ G(n) = \sum_{i=1}^n G(i-1) G(n-i) $$

- 状态转移方程 $G(n) = \sum_{i=1}^n G(i-1) G(n-i)$
- 初始值 G(0) = 1, G(1) = 1

```python
class Solution:
    def numTrees(self, n: int) -> int:
        g0 = 1
        g1 = 1
        if n == 0: return g0
        if n == 1: return g1
        G = [g0,g1] + [0] * (n-1)
        for j in range(2,n+1):
            for i in range(1,j+1):
                G[j] += G[i-1] * G[j-i]
        return G[-1]
```

#### [279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/)
1. 队列构造 BTS 广度优先搜索
2. 动态规划

![](assets/markdown-img-paste-20200103163948976.png)

```python
class Solution:
    def numSquares(self, n: int) -> int:
        from collections import deque
        queue = deque([n]) # 通过队列构造 BTS
        seen = set() # 如果之前见过，没必要再搜索一次
        level = 1
        while queue:
            # 遍历完同级后,level+1
            for _ in range(len(queue)):
                node_val = queue.popleft()
                for item in range(1, int(node_val**0.5)+1):
                    node = node_val - item**2
                    if node == 0: return level
                    if node not in seen:
                        queue.append(node)
                        seen.add(node)
            level += 1
        return level
```

#### [300. 最长上升子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)
时间复杂度O(n^2), O(n)
```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if len(nums) == 0: return 0
        dp = [1] * len(nums)
        for i in range(len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j]+1)
        return max(dp)
```
遍历nums，二分查找当前元素在dp中的low bound，替换dp中对应元素为当前元素，如果low bound 超过历史长度，长度+1. O(nlogn), O(n)
```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        """只能保证长度对，不能保证dp就是其中一个答案"""
        dp = [0] * len(nums)
        lenth = 0
        for num in nums:
            l, r = 0, lenth
            while (l < r):
                m = l + (r-l)//2
                if dp[m] < num: # <= 非严格上升子序列
                    l = m + 1
                else:
                    r = m
            if l < lenth:
                dp[l] = num
            else:
                dp[l] = num
                lenth += 1
        return lenth
```

#### [322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)
https://leetcode-cn.com/problems/coin-change/solution/dong-tai-gui-hua-tao-lu-xiang-jie-by-wei-lai-bu-ke/)
如果只是求最小个数，相当于问题只问了一半，可以用广度优先来做，但如果要列举所有满足条件的可能，还是需要动态规划或者递归来做，例如题377。
```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount+1)
        dp[0] = 0 # 注意初始化是0!
        for i in range(1, amount+1):
            for coin in coins:
                if i < coin:
                    dp[i] = dp[i]
                else:
                    dp[i] = min(dp[i], dp[i-coin]+1) # 注意是 dp[i-coin]+1
        return dp[-1] if dp[-1] != float('inf') else -1
```
```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        if amount == 0: return 0
        from collections import deque
        def bfs(amount, level):
            queue = deque([amount])
            seen = set([amount])
            level = 0
            while queue:
                for _ in range(len(queue)):
                    top = queue.pop()
                    for coin in coins:
                        res = top - coin
                        if res == 0:
                            return level+1
                        # 剪枝 important
                        if res > 0 and res not in seen:
                            seen.add(res)
                            queue.appendleft(res)
                level += 1
            return -1
        return bfs(amount, 0)
```

#### [面试题 08.11. 硬币](https://leetcode-cn.com/problems/coin-lcci/)
```python
import functools
class Solution:
    def waysToChange(self, n: int) -> int:
        coins = [25, 10, 5, 1]
        n_coin = len(coins)
        mod = 10**9 + 7
        @functools.lru_cache(None)
        def helper(cur_value, index):
            if cur_value < 0:
                return 0
            if cur_value == 0:
                return 1
            res = 0
            for i in range(index, n_coin):
                res += helper(cur_value-coins[i], i)
            return res
        return helper(n, index=0) % mod
    def waysToChange(self, n: int) -> int:
        mod = 10**9 + 7
        coins = [25, 10, 5, 1]

        f = [1] + [0] * n
        for coin in coins:
            for i in range(coin, n + 1):
                f[i] += f[i - coin]
        return f[n] % mod
```

#### [338. 比特位计数](https://leetcode-cn.com/problems/counting-bits/)
```python
class Solution:
    def countBits(self, num: int) -> List[int]:
        dp = [0] * (num+1)
        count = 0
        pivot = pow(2, count)
        for i in range(1, num+1):
            if i == pow(2, count+1):
                count += 1
                pivot = pow(2, count)

            dp[i] = 1 + dp[i-pivot]

        return dp
```

#### [416. 分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/)
```python
二维dp
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        nums_sum = sum(nums)
        if nums_sum % 2 != 0: return False
        target = nums_sum // 2

        dp = [[False] * (target+1) for _ in range(len(nums)+1)]
        dp[0][0] = True

        for i in range(1, len(dp)):
            for j in range(1, len(dp[0])):
                if j < nums[i-1]:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i-1]]

        return dp[-1][-1]
空间优化，不断覆盖之前的记录
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        if sum(nums) % 2 != 0: return False
        target = sum(nums) // 2
        dp = [False] * (target+1)
        dp[0] = True

        for num in nums:
            for i in range(target, num-1, -1):
                dp[i] = dp[i] or dp[i-num]

        return dp[-1]
```

#### [494. 目标和](https://leetcode-cn.com/problems/target-sum/)
```python
class Solution:
    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        if (S + sum(nums)) % 2 != 0 or sum(nums) < S: return 0
        T = (S + sum(nums)) // 2
        dp = [0] * (T+1)
        dp[0] = 1
        for num in nums:
            for j in range(T, num-1, -1): # 注意到 num-1，否则索引<0反向更新
                dp[j] = dp[j] + dp[j-num] # 不放num的方法数 + 放num之前容量的方法数
        return dp[-1]
```
#### [680. 验证回文字符串 Ⅱ](https://leetcode-cn.com/problems/valid-palindrome-ii/)
```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        def check(l, r):
            while l < r:
                if s[l] == s[r]:
                    l += 1
                    r -= 1
                else:
                    return False
            return True

        if s == s[::-1]: return True
        l, r = 0, len(s)-1
        while l < r:
            if s[l] == s[r]:
                l += 1
                r -= 1
            else:
                # if s[l+1:r+1] == s[l+1:r+1][::-1] or s[l:r] == s[l:r][::-1]:
                if check(l+1, r) or check(l, r-1):
                    return True
                else:
                    return False
        return True
```
#### [647. 回文子串](https://leetcode-cn.com/problems/palindromic-substrings/)
```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        # 中心拓展法
        count = len(s)
        if count <= 1: return count
        for i in range(len(s)):
            # 重点，两个回文中心
            j = 1
            while (i-j >= 0 and i+j < len(s) and s[i-j] == s[i+j]):
                count += 1
                j += 1
            j = 1
            while (i-j+1 >= 0 and i+j < len(s) and s[i-j+1] == s[i+j]):
                count += 1
                j += 1
        return count

class Solution:
    def countSubstrings(self, s: str) -> int:
        # 二维dp
        dp = [[0]*len(s) for _ in range(len(s))]
        for i in range(len(s)):
            dp[i][i] = 1
        for i in range(1, len(s)):
            for j in range(0, i):
                # 对角线旁的特殊处理
                if i-j == 1:
                    if s[i] == s[j]:
                        dp[i][j] = 1
                else:
                    if s[i] == s[j] and dp[i-1][j+1]:
                        dp[i][j] = 1
        count = 0
        for i in range(len(s)):
            count += sum(dp[i])
        return count

class Solution:
    def countSubstrings(self, s: str) -> int:
        # 一维dp
        dp = [0]*len(s)
        dp[0] = 0
        count = 0
        for i in range(1, len(s)):
            for j in range(0, i):
                # 对角线旁的特殊处理
                if i-j == 1:
                    if s[i] == s[j]:
                        dp[j] = 1
                    else: dp[j] = 0
                else:
                    if s[i] == s[j] and dp[j+1]:
                        dp[j] = 1
                    else: dp[j] = 0
            dp[i] = 1
            count += sum(dp)
        return count+1
```

[5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)
```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        # 1维dp
        if len(s) <= 1: return s
        dp = [0] * len(s)
        min_l = len(s)
        max_r = 0
        max_lenth = 0
        for r in range(1, len(s)):
            for l in range(r):
                if r - l == 1:
                    if s[r] == s[l]:
                        dp[l] = 1
                        if r-l > max_lenth:
                            max_lenth = r-l
                            min_l = l
                            max_r = r
                    else:
                        dp[l] = 0
                else:
                    if s[r] == s[l] and dp[l+1]:
                        dp[l] = 1
                        if r-l > max_lenth:
                            max_lenth = r-l
                            min_l = l
                            max_r = r
                    else:
                        dp[l] = 0
                dp[r] = 1

        return s[min_l:max_r+1] if max_lenth != 0 else s[0]
```

#### [121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        """profit_1记录数组最小值，profit_0 记录当前值与最小值的差"""
        if len(prices)==0: return 0
        profit_0 = 0
        profit_1 = -max(prices)
        for item in prices:
            profit_0 = max(profit_0, profit_1+item)
            profit_1 = max(profit_1, -item)
        return profit_0
```

## 贪心算法
在每一步选择中都采取在当前状态下最好或最优（即最有利）的选择，从而希望导致结果是最好或最优的算法,
贪心使用前提,局部最优可实现全局最优.
#### [406. 根据身高重建队列](https://leetcode-cn.com/problems/queue-reconstruction-by-height/)
```python
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        # 从大往小贪心排
        people = sorted(people, key=lambda ele: (-ele[0], ele[1]))
        result = []
        for item in people:
            index = item[1]
            result.insert(index,item)
        return result
```

#### [621. 任务调度器](https://leetcode-cn.com/problems/task-scheduler/)
```python
import collections

class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        dict_task = collections.Counter(tasks)
        time = 0
        while (max(dict_task.values()) > 0):
            count = 0
            for key in dict_task.most_common():
                if count < n+1:
                    if dict_task[key[0]] > 0:
                        dict_task[key[0]] -= 1
                        time += 1
                        count += 1
                else:
                    break
            if count < n + 1 and max(dict_task.values()) > 0:
                time += n + 1 - count
        return time
```

## 树
### 树的遍历
![例子](assets/leetcode-78b089f0.png)
- 深度遍历：`后序遍历, 前序遍历, 中序遍历`
- 广度遍历：`层次遍历`

#### [94. 二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)
输出顺序如同看二叉树的俯视图：左后 -> 中间节点 -> 右后。递归回溯

前提：任何一个节点都有左孩子，叶子左孩子为`None`
- 从该节点出发，一直递归到其最左节点
- 当该节点左孩子为`None`，该层递归退出，保存该节点
- 尝试去访问该节点右孩子，若为`None`则退出该层递归，返回并保存父节点
- 若不为`None`则去寻找该右孩子的最左节点

解法一： 递归
- 时间复杂度：O(n)。递归函数 T(n) = 2 * T(n/2) + 1。
- 空间复杂度：最坏情况下需要空间O(n)，平均情况为O(logn)
```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        def traversal(node, res):
            if node != None:
                traversal(node.left, res)
                res.append(node.val)
                traversal(node.right, res)

        res = []
        traversal(root, res)
        return res
```
遍历
```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        stack = []
        result = []
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            if stack:
                root = stack.pop()
                result.append(root.val)
                root = root.right
        return result
```

#### [144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)
输出顺序：根 -> 左子节点 -> 右子节点. dfs
思路：
- 从根节点开始，若当前节点非空，输出
- 依次向左，左子为空再向右

```python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        def traversal(node, res):
            if node != None:
                res.append(node.val)
                traversal(node.left, res)
                traversal(node.right, res)

        res = []
        traversal(root, res)
        return res
```

```python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        stack = []
        result = []
        while root or stack:
            while root:
                result.append(root.val)
                stack.append(root)
                root = root.left
            if stack:
                root = stack.pop()
                root = root.right
        return result
```

#### [145. 二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/)
输出顺序：左后 -> 右后 -> 根

```python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        def traversal(node, res):
            if node != None:
                traversal(node.left, res)
                traversal(node.right, res)
                res.append(node.val)

        res = []
        traversal(root, res)
        return res
```

```python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        if not root: return []
        stack = [root]
        result = []
        while stack:
            temp = stack.pop()
            if temp != None:
                stack.append(temp)
                stack.append(None)
                if temp.right:
                    stack.append(temp.right)
                if temp.left:
                    stack.append(temp.left)
            else:
                result.append(stack.pop().val)
        return result
```

#### [102. 二叉树的层次遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)
输出顺序：按层级从左到右. bfs
递归
```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        def traversal(node, level, res):
            if node != None:
                if len(res) == level: res.append([])
                res[level].append(node.val)
                traversal(node.left, level+1, res)
                traversal(node.right, level+1, res)

        res = []; level = 0
        traversal(root, level, res)
        return res
```
非递归
```python
from collections import deque
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root: return []
        result, level = [], 0
        queue = deque([root])
        while queue:
            result.append([])
            for i in range(len(queue)):
                top = queue.pop()
                result[level].append(top.val)
                if top.left:
                    queue.appendleft(top.left)
                if top.right:
                    queue.appendleft(top.right)
            level += 1
        return result
```

#### [542. 01 矩阵](https://leetcode-cn.com/problems/01-matrix/)
```python
from collections import deque
class Solution:
    def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        rows = len(matrix)
        if rows == 0:
            return []
        cols = len(matrix[0])

        def bfs(i, j):
            queue = deque([(i, j)])
            visited = set([(i, j)])
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            level = 0
            while queue:
                level += 1
                for _ in range(len(queue)):
                    row, col = queue.pop()
                    for direction in directions:
                        next_row = row + direction[0]
                        next_col = col + direction[1]
                        if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                            continue
                        if matrix[next_row][next_col] == 0:
                            return level
                        if matrix[next_row][next_col] == 1 and (next_row, next_col) not in visited:
                            queue.appendleft((next_row, next_col))
                            visited.add((next_row, next_col))


        result = [[0] * cols for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == 0:
                    result[i][j] = 0
                else:
                    result[i][j] = bfs(i, j)

        return result
```

#### [994. 腐烂的橘子](https://leetcode-cn.com/problems/rotting-oranges/)
坑很多。。
1. bfs 可以以多个节点为起始，不要被二叉树束缚
2. 注意已经访问过的节点设置为已访问
3. 返回level-1
3. 注意边界条件，左开右闭
4. 注意检查0， -1 的情况
```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        from collections import deque

        grid_h = len(grid)
        grid_w = len(grid[0]) if grid_h != 0 else 0
        if grid_h == 0 or grid_w == 0: return 0

        queue = deque()
        count_fresh = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 2:
                    queue.appendleft([i, j])
                if grid[i][j] == 1:
                    count_fresh += 1
        if count_fresh == 0: return 0

        level = 0
        while queue:
            for _ in range(len(queue)):
                i, j = queue.pop()
                if i+1 < grid_h and grid[i+1][j] == 1:
                    queue.appendleft([i+1, j])
                    grid[i+1][j] = 2
                if i-1 >= 0 and grid[i-1][j] == 1:
                    queue.appendleft([i-1, j])
                    grid[i-1][j] = 2
                if j+1 < grid_w and grid[i][j+1] == 1:
                    queue.appendleft([i, j+1])
                    grid[i][j+1] = 2
                if j-1 >= 0 and grid[i][j-1] == 1:
                    queue.appendleft([i, j-1])
                    grid[i][j-1] = 2
            level += 1

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    return -1

        return level-1
```

#### [LCP 09. 最小跳跃次数](https://leetcode-cn.com/problems/zui-xiao-tiao-yue-ci-shu/)
```python
from collections import deque
class Solution:
    def minJump(self, jump: List[int]) -> int:
        """
        1. visited 数组 比 set 快
        2. left_max = max(left_max, top) 比 left_max = top 快10倍
        """
        queue = deque([0])
        n = len(jump)
        visited = [0] * n
        level = 0
        left_max = 0
        while queue:
            level += 1
            for _ in range(len(queue)):
                top = queue.pop()
                # jump to right
                next_index = top + jump[top]
                if next_index < n:
                    if not visited[next_index]:
                        queue.appendleft(next_index)
                        visited[next_index] = 1
                else:
                    return level
                # jump to left
                for i in range(left_max+1, top):
                    if not visited[i]:
                        queue.appendleft(i)
                        visited[i] = 1
                left_max = max(left_max, top)
```

#### [864. 获取所有钥匙的最短路径](https://leetcode-cn.com/problems/shortest-path-to-get-all-keys/)
三维的bfs
```python
class Solution:
    def shortestPathAllKeys(self, grid: List[str]) -> int:
        mapping = {"a":0, "b":1, "c":2, "d":3, "e":4, "f":5}
        n = len(grid)
        if n == 0: return -1
        m = len(grid[0])
        nk = 0
        start = []
        for i in range(n):
            for j in range(m):
                cell = grid[i][j]
                if cell.islower():
                    nk |= (1<<mapping[cell])
                if cell == "@":
                    start = [i, j]
        visited = [[[0 for k in range(1<<len(mapping))] for i in range(m)] for j in range(n)]
        row, col, k = start[0], start[1], 0
        queue = collections.deque([(row, col, k)])
        orients = [[-1,0],[1,0],[0,-1],[0,1]]
        level = 0
        while queue:
            level += 1
            for _ in range(len(queue)):
                row, col, k = queue.pop()
                # print(grid[row][col])
                for orient in orients:
                    nxt_row, nxt_col = row + orient[0], col + orient[1]
                    nxt_k = k
                    # 越界
                    if nxt_row<0 or nxt_row>=n or nxt_col<0 or nxt_col>=m:
                        continue
                    cell = grid[nxt_row][nxt_col]
                    # 该状态访问过
                    if visited[nxt_row][nxt_col][nxt_k]:
                        continue
                    # 遇到墙
                    if cell == "#":
                        continue
                    # 遇到门,没相应的钥匙
                    if cell.isupper() and (1<<mapping[cell.lower()]) & nxt_k == 0:
                        continue
                    # 遇到钥匙
                    if cell.islower():
                        nxt_k |= (1<<mapping[cell]) # 重复没关系
                        if nxt_k == nk:
                            return level

                    visited[nxt_row][nxt_col][nxt_k] = 1
                    queue.appendleft((nxt_row, nxt_col, nxt_k))
        return -1
```

#### [987. 二叉树的垂序遍历](https://leetcode-cn.com/problems/vertical-order-traversal-of-a-binary-tree/submissions/)
输出顺序：左 -> 右， 上 -> 下
```python
class Solution:
    def verticalTraversal(self, root: TreeNode) -> List[List[int]]:
        def traversal(node, level, res, levels, deep=0):
            if node != None:
                # 层次遍历
                if level not in levels: levels.append(level); res[level] = []
                traversal(node.left, level-1, res, levels, deep = deep + 1)
                res[level].append([node.val, deep])
                traversal(node.right, level+1, res, levels, deep = deep +  1)

        res = {}; res_order = []; level = 0; levels = []; out = []
        traversal(root, level, res, levels)
        # 按宽度排序
        for key in sorted(res.keys()):
            res_order.append(res[key])
        # 按深度排序（同时保证同深度的，值小的在前）
        for item in res_order:
            item = sorted(item, key=lambda ele:(ele[1],ele[0]))
            out.append([i[0] for i in item])
        return out

class Solution:
    def verticalTraversal(self, root: TreeNode) -> List[List[int]]:
        def traversal(node, level, deep, res):
            if node:

                traversal(node.left, level+1, deep-1, res)
                res.append((node.val, deep, level))
                traversal(node.right, level+1, deep+1, res)

        level = 0
        deep = 0
        res = []
        traversal(root, level, deep, res)

        res_deep = sorted(res, key=lambda ele: ele[1])
        output = []
        deep_level = -1
        deep_last = None

        for item in res_deep:
            val, deep, level = item
            if deep_last != deep:
                output.append([])
                deep_level += 1

            output[deep_level].append((val, level))
            deep_last = deep

        out = []
        for i in range(len(output)):
            output[i] = sorted(output[i], key=lambda ele: (ele[1], ele[0]))
            out.append([])
            for item in output[i]:
                out[i].append(item[0])

        return out
```

#### [104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree)
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if root == None: return 0
        stack = []
        stack.append([1, root])
        depth = 0
        while (stack):
            curr_depth, top = stack.pop()
            left = top.left
            right = top.right
            depth = max(curr_depth, depth)
            if right: stack.append([curr_depth+1,right])
            if left: stack.append([curr_depth+1,left])
        return depth
```

#### [226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)
注意体会递归的逐步进入与退出，变量的生命周期
```python
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if root != None:
            left_node = self.invertTree(root.left)
            right_node = self.invertTree(root.right)
            root.left = right_node
            root.right = left_node
        return root

class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        def traversal(node):
            if node:
                traversal(node.left)
                traversal(node.right)
                temp = node.left
                node.left = node.right
                node.right = temp
        traversal(root)
        return root
```

#### [437. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/submissions/)
```python
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> int:
        def traversal(node, stack):
            if node:
                stack.append(node.val)
                cur = 0
                for i in range(len(stack)-1, -1, -1):
                    cur += stack[i]
                    if cur == sum: # 注意比较区别 sum(stack[i:]) == sum_
                        self.count += 1
                traversal(node.left, stack)
                traversal(node.right, stack)
                stack.pop() # 注意递归变量的生命周期

        self.count = 0; stack = [] # self.count与直接定义count的区别
        traversal(root, stack)
        return self.count
```

二叉搜索树具有以下性质：
- 如果节点的左子树不空，则左子树上所有结点的值均小于等于它的根结点的值；
- 如果节点的右子树不空，则右子树上所有结点的值均大于等于它的根结点的值；
- 任意节点的左、右子树也分别为二叉查找树；（二叉搜索树的定义是递归的二叉搜索树的定义是递归的）
- 没有键值相等的节点
- 中序遍历是升序

#### [538. 把二叉搜索树转换为累加树](https://leetcode-cn.com/problems/convert-bst-to-greater-tree/)
这道题对理解递归，回溯很有帮助。以树为例子，递归从root开始，在root结束。
递归回溯是一个不断深入，又回溯退出，在之间的操作注意理解同级性
```python
class Solution:
    def convertBST(self, root: TreeNode) -> TreeNode:
        def traversal(node):
            if node:
                traversal(node.right)
                node.val += self.last_value
                self.last_value = node.val
                traversal(node.left)

        self.last_value = 0
        traversal(root)
        return root
```

#### [543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree)
注意理解递归，通过dsf遍历得到每个当前节点的直径，保存最大直径
重点理解递归的 return, 二叉树遍历的退出,很好的练习

```python
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.max_diam = 0
        def traversal(node):
            # 递归到底部，返回基础值
            if node == None:
                return 0
            # 从底部归上来，每层如何处理，返回中间值
            else:
                L = traversal(node.left)
                R = traversal(node.right)
                self.max_diam = max(self.max_diam, L+R)
                return max(L, R) + 1
        _ = traversal(root)
        return self.max_diam
```

### 图
#### [399. 除法求值](https://leetcode-cn.com/problems/evaluate-division/)
```python
from collections import defaultdict, deque

class Solution:
    def bfs(self, query, graph):
            top, bottom = query
            visited = set([top])
            queue = deque([[top, 1]]) # careful
            while queue:
                top, value = queue.pop()
                if top == bottom:
                    return value
                for item in graph[top]:
                    if item not in visited:
                        visited.add(item)
                        queue.appendleft([item, value * graph[top][item]])
            return -1

    def dfs(self, query, graph):
        top, bottom = query
        visited = set([top])
        queue = deque([[top, 1]])
        while queue:
            top, value = queue.pop()
            if top == bottom:
                return value
            for item in graph[top]:
                if item not in visited:
                    visited.add(item)
                    queue.append([item, value * graph[top][item]])
        return -1

    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        graph = defaultdict(dict)
        chars = set()
        for equation, value in zip(equations, values):
            x, y = equation[0], equation[1]
            chars.update(equation)
            graph[x][y] = value
            graph[y][x] = 1 / value

        result = []
        for query in queries:
            value = -1 if query[0] not in chars and query[1] not in chars else self.dfs(query, graph)
            result.append(value)
        return result
```

[473. 火柴拼正方形](https://leetcode-cn.com/problems/matchsticks-to-square/)
```python
class Solution:
    def makesquare(self, nums: List[int]) -> bool:
        sum_val = sum(nums)
        if sum_val == 0 or sum_val % 4 != 0: return False
        target = sum_val // 4
        nums = sorted(nums, reverse=True)

        n = len(nums)
        visited = [0] * n
        def dfs(consum, cnt, index):
            if cnt == 4:
                return True
            if consum == target:
                return dfs(0, cnt+1, 0)
            if consum > target:
                return False
            i = index
            while i < n:
                if visited[i]:
                    i += 1
                    continue
                visited[i] = 1
                if dfs(consum+nums[i], cnt, i): return True
                # if seach fails, set visited back to 0
                visited[i] = 0
                # if dfs in first and last fails, return False
                if not consum or consum+nums[i] == target: return False
                # skip same num
                skip = i
                while skip < n and nums[skip] == nums[i]:
                    skip += 1
                i = skip
            return False

        return dfs(0,0,0)
```

#### [207. 课程表](https://leetcode-cn.com/problems/course-schedule)
```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        indegrees = [0 for _ in range(numCourses)]
        adjacency = [[] for _ in range(numCourses)]

        for item in prerequisites:
            curr, pre = item[0], item[1]
            adjacency[pre].append(curr)
            indegrees[curr] += 1
        queue = []
        for i, degree in enumerate(indegrees):
            if degree == 0:
                queue.append(i)
        while queue:
            pre = queue.pop()
            numCourses -= 1
            for curr in adjacency[pre]:
                indegrees[curr] -= 1
                if indegrees[curr] == 0:
                    queue.append(curr)

        return True if numCourses == 0 else False
```

#### [210. 课程表 II](https://leetcode-cn.com/problems/course-schedule-ii)
```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        adjacency = [[] for i in range(numCourses)]
        outdegree = [0 for i in range(numCourses)]
        for condition in prerequisites:
            curr, prev = condition
            adjacency[prev].append(curr)
            outdegree[curr] += 1
        stack = []
        for idx, item in enumerate(outdegree):
            if item == 0:
                stack.append(idx)

        results = []
        while stack:
            prev = stack.pop()
            numCourses -= 1
            results.append(prev)
            for curr in adjacency[prev]:
                outdegree[curr] -= 1
                if outdegree[curr] == 0:
                    stack.append(curr)
        return results if numCourses==0 else []
```
#### [1042. 不邻接植花](https://leetcode-cn.com/problems/flower-planting-with-no-adjacent)
```python
class Solution:
    def gardenNoAdj(self, N: int, paths: List[List[int]]) -> List[int]:
        adjacency = [[] for _ in range(N)]
        for path in paths:
            x, y = path[0]-1, path[1]-1
            adjacency[x].append(y)
            adjacency[y].append(x)
        result = [1] * N
        for i in range(N):
            flower = [1,2,3,4]
            for garden in adjacency[i]:
                if result[garden] in flower:
                    flower.remove(result[garden])
            result[i] = flower[0]
        return result
```
### 杂
#### [406. 根据身高重建队列](https://leetcode-cn.com/problems/queue-reconstruction-by-height/)
重点：高的可以忽视前面低的人
```python
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        # 从大往小贪心排
        people = sorted(people, key=lambda ele: (-ele[0], ele[1]))
        result = []
        for item in people:
            index = item[1]
            result.insert(index,item)
        return result
```

#### [58. 最后一个单词的长度](https://leetcode-cn.com/problems/length-of-last-word)
```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        l = 0
        flag = 0
        for i in s[::-1]:
            if not i.isspace():
                l += 1
                flag = 1
            if i.isspace() and flag: break
        return l

class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        split_list = s.split()
        if split_list:
            return len(split_list[-1])
        else: return 0
```
#### [1111. 有效括号的嵌套深度](https://leetcode-cn.com/problems/maximum-nesting-depth-of-two-valid-parentheses-strings/)
脑经急转弯
```python
class Solution:
    def maxDepthAfterSplit(self, seq: str) -> List[int]:
        ans = []
        depth = 0
        for item in seq:
            if item == "(":
                depth += 1
                ans.append(depth % 2)
            if item == ")":
                ans.append(depth % 2)
                depth -= 1

        return ans
```
#### [67. 二进制求和](https://leetcode-cn.com/problems/add-binary)
```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        grap = abs(len(a) - len(b))
        if len(a) > len(b):
            b = '0' * grap + b
        else: a = '0' * grap + a;
        s = ''
        add = 0
        for i in range(-1, -len(a)-1, -1):
            res = int(a[i]) + int(b[i]) + add
            add = 0
            if res > 1:
                res = res % 2
                add = 1
            s += str(res)
        if add == 1: s += str(1)
        return s[::-1]

class Solution:
    def addBinary(self, a: str, b: str) -> str:
        if len(a) > len(b):
            b = '0' * (len(a) - len(b)) + b
        else:
            a = '0' * (len(b) - len(a)) + a

        out = ''
        next_ = 0

        for i in range(len(a)-1,-1,-1):
            c = int(a[i]) + int(b[i]) + next_
            next_ = 0
            if c > 1:
                c -= 2
                next_ = 1
            out += str(c)

        if next_ == 1:
            out += '1'

        return out[::-1]
```
#### [66. 加一](https://leetcode-cn.com/problems/plus-one)
```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        if digits[-1] < 9:
            digits[-1] += 1
            return digits

        digits[-1] += 1
        for i in range(len(digits)-1, 0, -1):
            if digits[i] == 10:
                digits[i] = 0
                digits[i-1] += 1
        if digits[0] == 10:
            digits[0] = 0
            digits.insert(0,1)

        return digits
```

#### [283. 移动零](https://leetcode-cn.com/problems/move-zeroes)
```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        end = len(nums)
        i = 0
        while (i<end): # 注意inplace操作不要用for
            if nums[i] == 0:
                nums.pop(i)
                nums.append(0)
                end -= 1
            else:
                i += 1
```

#### [581. 最短无序连续子数组](https://leetcode-cn.com/problems/shortest-unsorted-continuous-subarray/)
```python
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        l, r = len(nums), 0
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if nums[i] > nums[j]:
                    l = min(l, i)
                    r = max(r, j)
        return r-l+1 if r-l+1 > 0 else 0

class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        sorted_nums = sorted(nums)
        l, r = len(nums), 0
        for i in range(len(nums)):
            if nums[i] != sorted_nums[i]:
                l = min(l, i)
                r = max(r, i)
        return r-l+1 if r-l+1 > 0 else 0
```
#### [560. 和为K的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/)
```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        # 最直接方法，O(n^2) 超时
        count = 0
        for i in range(len(nums)):
            sum_ = nums[i]
            if sum_ == k:
                count += 1

            for j in range(i+1, len(nums)):
                sum_ += nums[j]
                if sum_ == k:
                    count += 1

        return count

class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        # 如果累计总和，在索引i和j处相差k，即 sum[i] - sum[j] = k，则位于索引i和j之间的元素之和是k
        sum_dict = {}
        sum_dict[0] = 1
        sum_ = 0
        count = 0
        for item in nums:
            sum_ += item
            if sum_ - k in sum_dict:
                count += sum_dict[sum_-k]
            if sum_ not in sum_dict:
                sum_dict[sum_] = 1
            else:
                sum_dict[sum_] += 1
        return count
```

#### [445. 两数相加 II](https://leetcode-cn.com/problems/add-two-numbers-ii/)
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverse(self, head: ListNode):
        pre_node = None
        node = head
        while node:
            next_node = node.next
            node.next = pre_node
            pre_node = node
            node = next_node
        return pre_node

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        l1 = self.reverse(l1)
        l2 = self.reverse(l2)

        node1, node2 = l1, l2
        carry = 0
        ans = None
        while node1 or node2 or carry!=0:
            node1_val = node1.val if node1 else 0
            node2_val = node2.val if node2 else 0
            value = node1_val + node2_val + carry
            carry = value // 10
            node3 = ListNode(value%10)
            # 头插法
            node3.next = ans
            ans = node3
            if node1: node1 = node1.next
            if node2: node2 = node2.next

        return ans

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        s1, s2 = [], []
        while l1:
            s1.append(l1.val)
            l1 = l1.next
        while l2:
            s2.append(l2.val)
            l2 = l2.next
        ans = None
        carry = 0
        while s1 or s2 or carry != 0:
            a = 0 if not s1 else s1.pop()
            b = 0 if not s2 else s2.pop()
            cur = a + b + carry
            carry = cur // 10
            cur %= 10
            curnode = ListNode(cur)
            curnode.next = ans
            ans = curnode
        return ans
```

#### [83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if head == None: return None
        num_list = []
        node = head
        num_list.append(node.val)

        while(node.next):
            next_node = node.next
            if next_node.val not in num_list:
                num_list.append(next_node.val)
                node = node.next
            else: node.next = next_node.next
        return head

class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if head == None or head.next == None: return head
        node = head

        while(node.next):
            while(node.val == node.next.val):
                node.next = node.next.next
                if node.next == None: break
            if node.next == None: break
            else: node = node.next

        return head
```
#### [88. 合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/submissions/)
双指针
```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        nums1_copy = nums1[:m].copy()
        p0 = 0; p1 = 0; p3 = 0
        while (p0 < m and p1 < n):
            if nums1_copy[p0] < nums2[p1]:
                nums1[p3] = nums1_copy[p0]
                p0 += 1; p3 += 1
            else:
                nums1[p3] = nums2[p1]
                p1 += 1; p3 += 1
        if p0 == m: nums1[p3:] = nums2[p1:]
        else: nums1[p3:] = nums1_copy[p0:]
        return nums1

class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        i, j = 0, 0
        nums1[:] = nums1[:m]

        while (i < m and j < n):
            if nums1[i] > nums2[j]:
                nums1.insert(i, nums2[j]) # 注意insert后元素位置的变化, 数组大小的变化!
                j += 1
                i += 1
                m += 1
            else:
                i += 1

        if j < n: nums1.extend(nums2[j:])

        return nums1
```

#### [1296. 划分数组为连续数字的集合](https://leetcode-cn.com/problems/divide-array-in-sets-of-k-consecutive-numbers/)
```python
class Solution:
    def isPossibleDivide(self, nums: List[int], k: int) -> bool:
        dict_count = {}
        for i in range(len(nums)):
            if nums[i] not in dict_count:
                dict_count[nums[i]] = 1
            else:
                dict_count[nums[i]] += 1

        new_dict = {}
        for key in sorted(dict_count.keys()):
            new_dict[key] = dict_count[key]

        for key in new_dict:
            count = new_dict[key]
            if count > 0:
                try:
                    for i in range(key, key+k):
                        new_dict[i] = new_dict[i] - count
                except: return False

        for key in new_dict:
            if new_dict[key] != 0: return False
        return True
```


## 排序
排序算法测试
[912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)
### 比较排序
不稳定排序算法
堆排序,快速排序,选择排序,希尔排序
#### 快速排序
O(nlog(n)), 最坏 O(n^2)
```
快排的最差情况什么时候发生？
1. 已排序
2. 数值全部相等（已排序的特殊情况）

快排最好的情况是，每次正好中分，复杂度为O(nlogn)。最差情况，复杂度为O(n^2)，退化成冒泡排序
为了尽量避免最差情况的发生，就要尽量使每次选择的pivot为中位数。
一般常用的方法是，对每一个数列都取一次中位数(O(n))，这样总体的快排时间复杂度仍为O(nlogn)。
更为简化的方法是，取头、中、尾的中位数(O(1))作为pivot
```
1. 通过partition操作,使得pivot左边数均<pivot,右边>=pivot
2. 递归的对pivot左边,右边分别partition
3. 递归退出条件是l>=r
```python
def qsort(array, l, r):
    def partition(arr, left, right):
        pivot_val = arr[left]
        pivot_i = left
        for i in range(left+1, right):
            if arr[i] < pivot_val:
                pivot_i += 1
                arr[pivot_i], arr[i] = arr[i], arr[pivot_i]
        arr[pivot_i], arr[left] = arr[left], arr[pivot_i]
        return pivot_i

    if l < r:
        # partition: 交换，使得pivot左边<pivot,右边>=pivot
        pivot_index = partition_2(array, l, r)
        qsort(array, l, pivot_index)
        qsort(array, pivot_index+1, r)
```

中值快排: 解决的是复杂度退化到O(n^2)的问题
```python
def qsort(array, l, r):
    def get_median(l_i, r_i, m_i):
        l_val, r_val, m_val = nums[l_i], nums[r_i], nums[m_i]
        max_val = max(l_val, r_val, m_val)
        if l_val == max_val:
            mid_i = m_i if m_val > r_val else r_i
        elif r_val == max_val:
            mid_i = m_i if m_val > l_val else l_i
        else:
            mid_i = l_i if l_val > r_val else r_i
        return mid_i

    def partition(arr, left, right):
        m_i = left + (right-left)//2
        median_i = get_median(left, right-1, m_i)
        pivot_val = arr[median_i]
        arr[median_i], arr[left] = arr[left], arr[median_i]
        pivot_i = left
        for i in range(left+1, right):
            if arr[i] < pivot_val:
                pivot_i += 1
                arr[pivot_i], arr[i] = arr[i], arr[pivot_i]
        arr[pivot_i], arr[left] = arr[left], arr[pivot_i]
        return pivot_i

    if l < r:
        pivot_i = partition(array, l, r)
        qsort(l, pivot_i)
        qsort(pivot_i+1, r)
```

双路快排: 解决的是待排序数组中大量重复数字的问题
```python
def qsort(array, l, r):
    def partition2(arr, left, right):
        """双路快排，减少重复元素partition交换次数，无法解决退化n^2"""
        pivot = arr[left]
        l = left + 1
        r = right - 1
        while (l <= r): # 注意是 <= !
            # 左指针找到第一个大于pivot的数
            while (l < right and arr[l] <= pivot):
                l += 1
            # 右指针找到第一个小于pivot的数
            while (r > left and arr[r] >= pivot):
                r -= 1
            if l < r:
                arr[l], arr[r] = arr[r], arr[l]
        arr[left], arr[r] = arr[r], arr[left] # 注意是 r
        return r

    if l < r:
        # partition: 交换，使得pivot左边<pivot,右边>=pivot
        pivot_index = partition_2(array, l, r)
        qsort(array, l, pivot_index)
        qsort(array, pivot_index+1, r)
```

#### 归并排序
1. 递归对半分数组
2. 当被分子数组长度为1时,结束递归,return子数组
3. merge 返回的左右子数组
```python
def mergeSort(arr, l, r):
    def merge(l_arr, r_arr):
        result = []
        p1, p2 = 0, 0
        n1, n2 = len(l_arr), len(r_arr)
        while p1 < n1 and p2 < n2:
            if l_arr[p1] < r_arr[p2]:
                result.append(l_arr[p1])
                p1 += 1
            else:
                result.append(r_arr[p2])
                p2 += 1
        result.extend(l_arr[p1:] or r_arr[p2:])
        return result

    if r == 0:
        return []
    if l == r-1:
        return [arr[l]]
    m = l + (r-l)//2
    l_arr = mergeSort(arr, l, m)
    r_arr = mergeSort(arr, m, r)
    return merge(l_arr, r_arr)
```

#### 冒泡排序
O(n^2). 两两比较大小,每次循环将最大的数放在最后面
```python
def bubbleSort(array):
    n = len(array)
    for i in range(1, n):
        for j in range(n-i):
            if array[j+1] < array[j]:
                array[j], array[j+1] = array[j+1], array[j]
    return array
```

#### 选择排序
第二层循环,找到最小的数,放在最前面.O(n^2)复杂度,不受数组初始排序影响.
```python
def selectSort(array):
    n = len(array)
    for i in range(n):
        min_index = i
        for j in range(i+1, n):
            if array[j] < array[min_index]:
                min_index = j
        if min_index != i:
            array[i], array[min_index] = array[min_index], array[i]
    return array
```

#### 插入排序
把当前数,和前面的数比大小,赋值交换找到插入位置
```python
def insertionSort(arr):
    for i in range(len(arr)):
        preIndex = i - 1
        current = arr[i]
        while preIndex >= 0 and arr[preIndex] > current:
            arr[preIndex+1] = arr[preIndex]
            preIndex -= 1
        arr[preIndex+1] = current
    return arr
```

#### 堆排序
两次sift_down操作,第一次从倒数第二层到根节点下沉,保证根节点最大
第二次for循环把最大值交换到尾部,然后根下沉.
```python
def heapSort(arr):
    def sift_down(arr, root, k):
        root_val = arr[root] # 用插入排序的赋值交换
        # 确保交换后，对后续子节点无影响
        while (2*root+1 < k):
            # 构造根节点与左右子节点
            child = 2 * root + 1  # left = 2 * i + 1, right = 2 * i + 2
            if child+1 < k and arr[child] < arr[child+1]: # 如果右子节点在范围内且大于左节点
                child += 1
            if root_val < arr[child]:
                arr[root] = arr[child]
                root = child
            else: break # 如果有序，后续子节点就不用再检查了
        arr[root] = root_val

    n = len(arr) # n 为heap的规模
    # 保证根节点最大. 从倒数第二层向上，该元素下沉
    for i in range((n-1)//2, -1, -1):
        sift_down(arr, i, n)
    # 从尾部起，依次与顶点交换并再构造 maxheap，heap规模-1
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # 交换
        sift_down(arr, 0, i)
```

#### 希尔排序
TODO: CHECK!
```python
count = 1
inc = 2
while (inc > 1):
    inc = len(array) // (2 * count)
    count += 1
    for i in range(len(array)-inc):
        if array[i] > array[i+inc]: array[i+inc], array[i] = array[i], array[i+inc]
```

### 非比较排序
#### 计数排序
时间复杂度为O(n+k)，空间复杂度为O(n+k)。n 是待排序数组长度，k 是 max_value-min_value+1长度。
排序算法，即排序后的相同值的元素原有的相对位置不会发生改变。

可以排序整数（包括负数），不能排序小数
1. 计算数组值最大与最小，生成长度为 max-min+1 的bucket
2. 遍历待排序数组，将当前元素值-min作为index，放在bucket数组
3. 清空原数组，遍历bucket，原数组依次append
```python
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
```

#### 桶排序
桶排序是计数排序的拓展
![](assets/leetcode-be66e5dc.png)
如果对每个桶（共M个）中的元素排序使用的算法是插入排序，每次排序的时间复杂度为O(N/Mlog(N/M))。
则总的时间复杂度为O(N)+O(M)O(N/Mlog(N/M)) = O(N+ Nlog(N/M)) = O(N + NlogN - NlogM)。
当M接近于N时，桶排序的时间复杂度就可以近似认为是O(N)的。是一种排序算法.

可以排序负数与小数
```python
def bucketSort(array, n):
    min_value = min(array)
    max_value = max(array)
    bucket_count = int((max_value - min_value) / n) + 1
    buckets = [[] for _ in range(bucket_count)]
    for num in array:
        bucket_index = int((num - min_value) // n)
        buckets[bucket_index].append(num)
    array.clear()
    for bucket in buckets:
        insertionSort(bucket)
        for item in bucket:
            array.append(item)
```

#### 基数排序
非负整数排序
```python
def radixSort(array):
    rounds = len(str(max(array)))
    radix = 10
    for i in range(rounds):
        buckets = [[] for _ in range(radix)]
        for num in array:
            index = num // (10**i) % radix
            buckets[index].append(num)
        array.clear()
        for bucket in buckets:
            for item in bucket:
                array.append(item)
```


## 二分查找
### 基础 (前提，数组有序)
```python
def low_bound(arr, l, r, target):
    """查找第一个 >= target的数的index"""
    while (l < r):
        m = l + (r-l)//2
        if arr[m] < target:
            l = m + 1
        else:
            r = m
    return l

def up_bound(arr, l, r, target):
    """查找第一个 > target的数的index"""
    while (l < r):
        m = l + (r-l)//2
        if arr[m] <= target:
            l = m + 1
        else:
            r = m
    return l

index = low_bound(result, 0, len(result), array[i])
```

#### [LCP 08. 剧情触发时间](https://leetcode-cn.com/problems/ju-qing-hong-fa-shi-jian/)
```python
class Solution:
    def getTriggerTime(self, increase: List[List[int]], requirements: List[List[int]]) -> List[int]:
        """前缀和+二分"""
        n_increase = len(increase) + 1
        pre_sum_C = [0] * n_increase
        pre_sum_R = [0] * n_increase
        pre_sum_H = [0] * n_increase
        for i in range(1, n_increase):
            pre_sum_C[i] = pre_sum_C[i-1] + increase[i-1][0]
            pre_sum_R[i] = pre_sum_R[i-1] + increase[i-1][1]
            pre_sum_H[i] = pre_sum_H[i-1] + increase[i-1][2]

        def low_bound(arr, l, r, target):
            while (l < r):
                m = l + (r-l) // 2
                if arr[m] < target:
                    l = m + 1
                else:
                    r = m
            return l
        result = []
        for i in range(len(requirements)):
            min_C = low_bound(pre_sum_C, 0, n_increase, requirements[i][0])
            min_R = low_bound(pre_sum_R, 0, n_increase, requirements[i][1])
            min_H = low_bound(pre_sum_H, 0, n_increase, requirements[i][2])
            activate = max(min_C, min_R, min_H)
            res = -1 if activate >= n_increase else activate
            result.append(res)
        return result
```
### 二分估计查找
下面两题用的是二分估计查找的思路，数组并不有序，但是可以通过mid去计算基于mid下k,m的估计值，与实际值比较，
收紧区间，达到查找的目的。典型的特点是，[left,right]是值区间，而不是index区间。
####　[668. 乘法表中第k小的数](https://leetcode-cn.com/problems/kth-smallest-number-in-multiplication-table/)
这题没想到可以用二分，加了个判断可以快很多。 mid // n 可以定位mid所在行之前的行数，计数count += mid//n * n , 然后从mid//n + 1 开始遍历即可
```python
class Solution:
    def findKthNumber(self, m: int, n: int, k: int) -> int:
        left, right = 1, m*n
        while left < right:
            mid = left + (right-left)//2
            count = 0
            # 减少遍历次数
            start = mid // n
            count += start * n
            for i in range(start+1, m+1):
                # 统计的个数不能超过范围n,所以取min
                # count += min(mid // i, n)
                count += mid//i
            if count < k:
                left = mid + 1
            else:
                right = mid
        return left
```

#### [LCP 12. 小张刷题计划](https://leetcode-cn.com/problems/xiao-zhang-shua-ti-ji-hua/)
```python
class Solution:
    def minTime(self, time: List[int], m: int) -> int:
        if m >= len(time):
            return 0
        def check(mid, m):
            """if can fill m arrs, return True
               else return False"""
            prefix = 0
            max_time = 0
            for num in time:
                max_time = max(max_time, num)
                prefix += num
                if prefix - max_time > mid:
                    m -= 1
                    prefix = num
                    max_time = num # becareful
                    if m == 0:
                        return True
            return False

        low_bound, up_bound = min(time), sum(time)
        while low_bound < up_bound:
            mid = low_bound + (up_bound - low_bound) // 2
            if check(mid, m):
                low_bound = mid + 1
            else:
                up_bound = mid
        return low_bound
```

#### [69. x 的平方根](https://leetcode-cn.com/problems/sqrtx/)
```python
class Solution:
    def mySqrt(self, ａ: int) -> int:
        """牛顿法,解　f(x)-a=0 这个方程的正根
        核心: x' = x - f(x)/f'(x)
             if abs(x'-x)<1e-4: return x
        """
        if a == 0:
            return 0
        curr = a
        while True:
            prev = curr
            curr = (curr + a/curr) / 2
            if abs(curr - prev) < 1e-1:
                return int(curr)
        """二分法,小心边界"""
        left, right = 0, x
        while left < right:
            mid = left + (right-left+1)//2
            val = mid ** 2
            if val <= x:
                left = mid
            else:
                right = mid - 1
        return left
```
#### [441. 排列硬币](https://leetcode-cn.com/problems/arranging-coins/solution/er-fen-fa-by-xxinjiee/)
可以直接用数学公式求解，也可以通过二分法求解数学公式 类似[69. x的平方根](https://leetcode-cn.com/problems/sqrtx/)

这里使用二分查找求解的核心是
1. 定义左右边界，r 初始值限定为 n // 2 + 1，缩小范围
2. m为层数，循环中每次用l, r的中点更新
3. 定义target = m * (m + 1) / 2 待求解公式
4. 如果target < n - m (m 同时也是最后一层的个数)，更新查找范围下限l
5. 否则更新查找范围上限r，最后r = l 退出while loop，返回其中一个即可

```python
class Solution:
    def arrangeCoins(self, n: int) -> int:
        # 解方程 m(m+1) / 2 = n
        l = 0; r = n // 2 + 1
        while(l < r):
            m = l + (r - l) // 2
            target = m * (m + 1) / 2
            if target < n - m: l = m + 1
            else: r = m
        return l
```

附上二分查找的low_bound(),该题的主要区别就是定义target，替换low_bound()中的array[m]与被查找值的比较

```python
def low_bound(array, l, r, o):
    # 返回区间内第一个 >= o 的值, o 为被查找值
    while l < r:
        m = l + (r - l) // 2
        # l, r 的赋值规则也要符合左闭右开
        if array[m] < o: l = m + 1
        else: r = m
```

#### [33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)
在两个排序数组中使用二分搜索查找， 注意区间缩小的判断
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums)
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[left] < nums[mid]:
                if nums[mid] < target < nums[left]:
                    left = mid + 1
                else:
                    right = mid
            else:
                if nums[mid] < target < nums[left]:
                    left = mid + 1
                else:
                    right = mid
        return -1
```

#### [81. 搜索旋转排序数组 II](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/)
核心 if nums[mid] == nums[left]: left += 1
```python
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        left, right = 0, len(nums)
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return True
            elif nums[mid] == nums[left]:
                left += 1
            elif nums[mid] < nums[left]:
                if nums[mid] < target:
                    if nums[left] <= target:
                        right = mid
                    else:
                        left = mid + 1
                else:
                    right = mid
            else:
                if nums[mid] < target:
                    left = mid + 1
                else:
                    if nums[left] <= target:
                        right = mid
                    else:
                        left = mid + 1
        return False
```
#### [153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)
注意将mid前置，以right来判断
```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        # right-1 将mid前置
        left, right = 0, len(nums)-1
        while left < right:
            mid = left + (right - left) // 2
            # 必须与right比较，因为left=mid+1总是比right
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid
        return nums[left]
```

#### [154. 寻找旋转排序数组中的最小值 II](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/)
核心 if nums[mid] == nums[right]: right = right - 1
```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums)-1
        while left < right:
            mid = left + (right-left)//2
            if nums[mid] > nums[right]:
                left = mid + 1
            elif nums[mid] < nums[right]:
                right = mid
            else:
                right = right - 1
        return nums[left]
```

## 字符串
### 前缀树
[208. 实现 Trie (前缀树)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)
key是字符，value是node， class node 基本是个字典，有着判断是否结束的属性
```python
class Node:
    def __init__(self):
        self.is_end = False
        self.dict = {}

class Trie:
    def __init__(self):
        self.root = Node()

    def insert(self, word: str) -> None:
        cur_node = self.root
        for alpha in word:
            if alpha not in cur_node.dict:
                cur_node.dict[alpha] = Node()
            cur_node = cur_node.dict[alpha]
        cur_node.is_end = True

    def search(self, word: str) -> bool:
        cur_node = self.root
        for alpha in word:
            if alpha not in cur_node.dict:
                return False
            cur_node = cur_node.dict[alpha]
        return cur_node.is_end


    def startsWith(self, prefix: str) -> bool:
        cur_node = self.root
        for alpha in prefix:
            if alpha not in cur_node.dict:
                return False
            else:
                cur_node = cur_node.dict[alpha]
        return True
```
#### [211. 添加与搜索单词 - 数据结构设计](https://leetcode-cn.com/problems/add-and-search-word-data-structure-design/)
注意体会递归的设计, 挺多坑的
```python
class Node:
    def __init__(self):
        self.dict = {}
        self.is_end = False

class WordDictionary:
    def __init__(self):
        self.root = Node()

    def addWord(self, word: str) -> None:
        cur_node = self.root
        for alpha in word:
            if alpha not in cur_node.dict:
                cur_node.dict[alpha] = Node()
            cur_node = cur_node.dict[alpha]
        if not cur_node.is_end:
            cur_node.is_end = True

    def search(self, word: str) -> bool:
        return self.helper(self.root, 0, word)

    def helper(self, cur_node, i, word):
        if i == len(word):
            return cur_node.is_end # if no more in word
        if word[i] != '.':
            if word[i] not in cur_node.dict:
                return False
            return self.helper(cur_node.dict[word[i]], i+1, word)

        else:
            for key in cur_node.dict:
                if self.helper(cur_node.dict[key], i+1, word) == True:
                    return True # be careful, don't return False
            return False # if no more in trie
```

#### [212. 单词搜索 II](https://leetcode-cn.com/problems/word-search-ii/)
这道题整体思路是 1. 构建words的字典树 trie  2. 在board上深度优先遍历
但具体实现上，很多坑，在这里总结一下,好好体会递归
1. trie 中 在board上找到的单词结尾要设成已访问，保证结果无重复
2. 在递归进入board下一个节点的前，要把当前节点设成已访问，不然未来可能重复访问该节点
3. 基于当前节点的深度优先搜索结束后，恢复board当前节点的值，便于之后单词的搜索

```python
class Node:
    def __init__(self):
        self.dict = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = Node()

    def insert(self, word):
        cur_node = self.root
        for alpha in word:
            if alpha not in cur_node.dict:
                cur_node.dict[alpha] = Node()
            cur_node = cur_node.dict[alpha]
        cur_node.is_end = True


class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        self.h = len(board)
        self.w = len(board[0])
        self.res = []
        trie = Trie()
        for word in words:
            trie.insert(word)
        for i in range(len(board)):
            for j in range(len(board[0])):
                cur_node = trie.root
                self.dfs(i, j, cur_node, board, "")
        return self.res

    def dfs(self, i, j, cur_node, board, word=""):
        char = board[i][j]
        if char in cur_node.dict:
            word = word + char
            if cur_node.dict[char].is_end:
                self.res.append(word)
                cur_node.dict[char].is_end = False # 保证无重复

            cur_node = cur_node.dict[char]

            board[i][j] = None # 关键！每个单词，不走回头路

            if i+1 < self.h and board[i+1][j]!=None:
                self.dfs(i+1, j, cur_node, board, word)
            if i > 0 and board[i-1][j]!=None:
                self.dfs(i-1, j, cur_node, board, word)
            if j+1 < self.w and board[i][j+1]!=None:
                self.dfs(i, j+1, cur_node, board, word)
            if j > 0 and board[i][j-1]!=None:
                self.dfs(i, j-1, cur_node, board, word)

            board[i][j] = char # 关键！在内存中恢复board
```

#### [443. 压缩字符串](https://leetcode-cn.com/problems/string-compression/)
```python
class Solution:
    def compress(self, chars: List[str]) -> int:
        count = 1
        temp = chars[0]
        read = 1
        while (read < len(chars)):
            if chars[read] == temp:
                count += 1
                chars.pop(read)
                read -= 1
            else:
                if count != 1:
                    for item in str(count):
                        chars.insert(read, item)
                        read += 1
                count = 1
                temp = chars[read]
            read += 1
        if count != 1:
            for item in str(count):
                chars.insert(read, item)
                read += 1
        return len(chars)

class Solution:
    def compress(self, chars: List[str]) -> int:
        if len(chars) <= 1: return len(chars)
        count = 1
        pointer = 0
        dynamic_boundary = len(chars)
        while (pointer < dynamic_boundary-1):
            next_char = chars[pointer+1]
            curr_char = chars[pointer]
            if next_char == curr_char:
                count += 1
                chars.pop(pointer)
                dynamic_boundary -= 1
            else:
                pointer += 1
                if count > 1:
                    for item in str(count):
                        chars.insert(pointer, item)
                        dynamic_boundary += 1
                        pointer += 1
                    count = 1
        if count > 1:
            for item in str(count):
                chars.insert(pointer+1, item)
                pointer += 1
        return len(chars)
```
#### [541. 反转字符串 II](https://leetcode-cn.com/problems/reverse-string-ii/)
python字符串修改及其麻烦，转换成list，最后再通过''.join()转成str
```python
class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        pointer = 0
        s = list(s)
        while (pointer < len(s)):
            if pointer + k <= len(s):
                s[pointer:pointer+k] = s[pointer:pointer+k][::-1]
            else:
                s[pointer:] = s[pointer:][::-1]
            pointer += 2 * k
        s = ''.join(s)
        return s

class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        s_list = list(s)
        for i in range(len(s_list)):
            if i % (2*k) == 0:
                try:
                    s_list[i:i+k] = s_list[i:i+k][::-1]
                except:
                    s_list[i:] = s_list[i:][::-1]
        s_reverse = ''.join(s_list)
        return s_reverse
```

## 滑动窗口
先考虑双指针构成的list窗口能不能求解，再考虑把窗口化为字典
#### [76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring)
双指针，滑动窗口求解
TODO： 整理一下
```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        window = {}
        # 构造 window dict
        for item in t:
            if item not in window:
                window[item] = 1
            else: window[item] += 1

        # 根据 window 初始化 search_area dict
        search_area = {}
        for key in window.keys():
            search_area[key] = 0

        l_pointer, r_pointer = 0, 0
        ans, ans_len = '', len(s)

        while (r_pointer < len(s)):
            if s[r_pointer] in search_area.keys():
                search_area[s[r_pointer]] += 1
            r_pointer += 1 # 右指针右移
            self.flag = 1
            for key in window.keys():
                if search_area[key] < window[key]:
                    self.flag = 0
                    break
            # 如果search_area已经覆盖window,对search_area进行优化，移动左指针
            while self.flag:
                if s[l_pointer] in search_area.keys():
                    search_area[s[l_pointer]] -= 1
                    if search_area[s[l_pointer]] < window[s[l_pointer]]:
                        if len(s[l_pointer:r_pointer]) <= ans_len:
                            ans = s[l_pointer:r_pointer] #用ans记录当前符合条件的最小长度子串
                            ans_len = len(ans)
                        l_pointer += 1
                        break
                l_pointer += 1

        return ans

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        T_dict = {}
        for item in t:
            if item in T_dict:
                T_dict[item] += 1
            else:
                T_dict[item] = 1

        p1 = 0
        p2 = 0
        result = ''
        last_len = len(s)

        while (p2 < len(s)):
            if s[p2] in T_dict:
                T_dict[s[p2]] -= 1
            p2 += 1

            while max(T_dict.values()) <= 0:
                s_len = p2 - p1
                if s_len <= last_len:
                    result = s[p1:p2]
                    last_len = s_len
                if s[p1] in T_dict:
                    if T_dict[s[p1]] + 1 <= 0:
                        T_dict[s[p1]] += 1
                        p1 += 1
                    else: break
                else:
                    p1 += 1

        return result

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        from collections import Counter
        dict_t = Counter(t)
        require = len(dict_t)
        contained = 0
        l, r = 0, 0
        min_len = len(s)
        start, end = 0, 0

        while (r < len(s)):
            if s[r] in dict_t:
                dict_t[s[r]] -= 1
                if dict_t[s[r]] == 0:
                    contained += 1
            r += 1
            while (l<r and contained == require):
                if r-l <= min_len:
                    min_len = r-l
                    start = l
                    end = r
                if s[l] in dict_t:
                    dict_t[s[l]] += 1
                    if dict_t[s[l]] > 0:
                        contained -= 1
                l += 1

        if start == 0 and end == 0: return ""
        return s[start:end]
```

#### [30. 串联所有单词的子串](https://leetcode-cn.com/problems/substring-with-concatenation-of-all-words/)
words 构造一个字典， 遍历s，以word的长度，构造list进而构造字典，比较与words_dict是否相等
```python
class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        from collections import Counter
        if len(words) == 0: return []
        word_len = len(words[0])
        words_len = len(words) * word_len
        words_dict = Counter(words)
        l, r = 0, words_len
        start = 0
        result = []
        while (start+words_len <= len(s)):
            s_subs = [s[start+i*word_len:start+(i+1)*word_len] for i in range(len(words))]
            s_dict = Counter(s_subs)
            if s_dict == words_dict:
                result.append(start)
            start += 1
        return result
```

#### [438. 找到字符串中所有字母异位词](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string)、
方法一，通过双重for找到第一个满足的后，往后依次遍历。最直观但是超时
```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        # TODO： 超时方法 ！！
        p_dict = {}
        for item in p:
            if item in p_dict:
                p_dict[item] += 1
            else:
                p_dict[item] = 1

        p1 = 0
        result = []
        temp_dict = p_dict.copy() # 共享内存,修改temp,p_dict也跟着变!
        while (p1 < len(s) - len(p) + 1):
            if len(s) - p1 >= len(p):
                if s[p1] in temp_dict:
                    temp_dict[s[p1]] -= 1
                    p2 = p1 + 1
                    for i in range(p2, p2+len(p)-1):
                        if s[i] in temp_dict:
                            temp_dict[s[i]] -= 1
                            if min(temp_dict.values()) < 0:
                                break
                        else:
                            break
                    if max(temp_dict.values()) == 0:
                        result.append(p1)
                    temp_dict = p_dict.copy()
                p1 += 1
            else: break
        return result
```
方法二三，构造被检索子串字典，构造临时字典，双指针维护临时字典.
```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        res = []
        window = {}
        for item in p:
            if item not in window:
                window[item] = 1
            else: window[item] += 1

        search_area = {}
        for key in window.keys():
            search_area[key] = 0

        pointer = 0
        while (pointer < len(s)):
            self.flag = 1 # 注意flag初始化位置
            if s[pointer] in window.keys():
                search_area[s[pointer]] += 1
            for key in window.keys():
                if search_area[key] != window[key]:
                    self.flag = 0
                    break
            if self.flag: res.append(pointer - len(p) + 1)
            pointer += 1
            if pointer > len(p) - 1:
                if s[pointer-len(p)] in search_area.keys():
                    search_area[s[pointer-len(p)]] -= 1
        return res

class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        p_dict = {}
        for item in p:
            if item in p_dict:
                p_dict[item] += 1
            else:
                p_dict[item] = 1

        p1 = 0
        result = []
        temp_dict = {}
        for key in p_dict:
            temp_dict[key] = 0
        while (p1 < len(s)):
            if s[p1] in temp_dict:
                temp_dict[s[p1]] += 1
            flag = True
            for key in temp_dict:
                if temp_dict[key] != p_dict[key]:
                    flag = False
            if flag:
                result.append(p1 - len(p) + 1)
            p1 += 1

            if p1 > len(p) - 1:
                if s[p1-len(p)] in temp_dict:
                    temp_dict[s[p1-len(p)]] -= 1

        if temp_dict == p_dict:
            result.append(p1 - len(p) + 1)

        return result
```
#### [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters)
滑动窗口双指针，r向前，遇到重复字符后，l向前，过程中记录最大长度
```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        l, r, max_len = 0, 0, 0
        while (r < len(s)):
            if s[r] in s[l:r]:
                l += 1
            else:
                max_len = max(max_len, r-l+1)
                r += 1
        return max_len
```

#### [面试题57 - II. 和为s的连续正数序列](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)
这题也能滑动窗口，构造1-target的list，sum[l:r]<target, r向前走，sum[l:r]>target, l向前走， sum[l:r]>target，记录，l向前走
```python
class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        """滑动窗口"""
        target_list = [i+1 for i in range(target)]
        l, r = 0, 1
        result = []
        while (r < len(target_list)):
            if sum(target_list[l:r]) < target:
                r += 1
            elif sum(target_list[l:r]) > target:
                l += 1
            else:
                result.append([i for i in target_list[l:r]])
                l += 1 # important
        return result
```

## 递归算法复杂度分析 -- 主定理
T(问题规模) = 子问题数 * T(子问题规模) + 额外计算
T(n) = a * T(n/b) + f(n)
T(n) = a * T(n/b) + O(n^d)
- $d < log_b^a, O(n^{log_b^a})$
- $d = log_b^a, O(n^d * logn)$
- $d > log_b^a, O(n^d)$
(log 不标底默认为2)

归并排序
T(n) = 2T(n/2) + O(n) --> T(n) = O(nlogn)

二分查找
T(n) = T(n/2) + O(1) --> T(n) = O(logn)

#### [56. 合并区间](https://leetcode-cn.com/problems/merge-intervals/)
```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if len(intervals) <= 1: return intervals
        result = []
        intervals = sorted(intervals, key=lambda ele: (ele[0]))
        is_not_end = True
        index = 0
        while is_not_end:
            if intervals[index][1] >= intervals[index+1][0]:
                up_bound = max(intervals[index][1], intervals[index+1][1])
                low_bound = min(intervals[index][0], intervals[index+1][0])
                intervals[index+1] = [low_bound, up_bound]
                intervals.pop(index)
            else:
                index += 1
            if index+1 >= len(intervals):
                is_not_end = False
        return intervals

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])
          merged = []
          for interval in intervals:
              # 如果列表为空，或者当前区间与上一区间不重合，直接添加
              if not merged or merged[-1][1] < interval[0]:
                  merged.append(interval)
              else:
                  # 否则的话，我们就可以与上一区间进行合并
                  merged[-1][1] = max(merged[-1][1], interval[1])
          return merged
```
###
```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        """O(logn)"""
        if x == 0: return 0
        if n < 0:
            n, x = -n, 1/x
        res = 1
        while n > 0:
            if n & 1:
                res *= x
            x *= x
            n = n >> 1
        return res
```
### 树状数组
```python
class FenwickTree:
    def __init__(self, n):
        self.size = n
        self.tree = [0 for _ in range(n+1)]

    def lowbit(self, index):
        """算出x二进制的从右往左出现第一个1以及这个1之后的那些0组成数的二进制对应的十进制的数.以88为例, 88 = 1011000, 第一个1以及他后面的0组成的二进制是1000,对应的十进制是8，所以c一共管理8个a。
        """
        return index & (-index)

    def update(self, index, delta):
        while index <= self.size:
            self.tree[index] += delta
            index += self.lowbit(index)

    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= self.lowbit(index)
        return res
```
### 位操作
python 中 bin 可以十进制转二进制。二进制"0b"，八进制"0"，十六进制"0x"开头。
位运算说明
```python
x >> y # x 右移 y 位
x << y # x 左移 y 位
x & y # 只有 1 and 1 = 1，其他情况位0
x | y # 只有 0 or 0 = 0，其他情况位1
~x # 反转操作，对 x 求的每一位求补，结果是 -x - 1
x ^ y # 或非运算，如果 y 对应位是0，那么结果位取 x 的对应位，如果 y 对应位是1，取 x 对应位的补
```

向右移1位可以看成除以2，向左移一位可以看成乘以2。移动n位可以看成乘以或者除以2的n次方。
```python
8 >> 2 <=> 8 / 2 / 2 <=> 0b1000 >> 2 = 0b10 = 2
8 << 2 <=> 8 * 2 * 2 <=> 0b1000 << 2 = 0b100000 = 32
```

#### [318. 最大单词长度乘积](https://leetcode-cn.com/problems/maximum-product-of-word-lengths/)
单词仅包含小写字母，可以使用 26 个字母的位掩码对单词的每个字母处理，判断是否存在某个字母。如果单词中存在字母 a，则将位掩码的第一位设为 1，否则设为 0。如果单词中存在字母 b，则将位掩码的第二位设为 1，否则设为 0。依次类推，一直判断到字母 z。
```python
class Solution:
    def maxProduct(self, words: List[str]) -> int:
        n = len(words)
        masks = [0] * n
        lens = [0] * n
        bit_number = lambda ch : ord(ch) - ord('a')

        for i in range(n):
            bitmask = 0
            for ch in words[i]:
                # 将字母对应位设置为1
                bitmask |= 1 << bit_number(ch)
            masks[i] = bitmask
            lens[i] = len(words[i])

        max_val = 0
        for i in range(n):
            for j in range(i + 1, n):
                if masks[i] & masks[j] == 0:
                    max_val = max(max_val, lens[i] * lens[j])
        return max_val
```

#### [983. 最低票价](https://leetcode-cn.com/problems/minimum-cost-for-tickets/)
动态规划，dp 长度为days[-1]+1, 值为0，对于days里的每一天，状态只可能从1，7，30天前转移过来。
在三种状态下取最小的cost即可
```python
class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        dp = [0] * (days[-1]+1)
        day_index = 0
        for i in range(days[-1]):
            i += 1
            if i != days[day_index]:
                dp[i] = dp[i-1]
                continue
            else:
                day_index += 1
                dp[i] = min(
                            dp[max(0,i-1)]+costs[0],
                            dp[max(0,i-7)]+costs[1],
                            dp[max(0,i-30)]+costs[2])
        return dp[-1]
```

#### [5409. 检查一个字符串是否包含所有长度为 K 的二进制子串](https://leetcode-cn.com/problems/check-if-a-string-contains-all-binary-codes-of-size-k/)
```
输入：s = "00110110", k = 2  输出：true
解释：长度为 2 的二进制串包括 "00"，"01"，"10" 和 "11"。它们分别是 s 中下标为 0，1，3，2 开始的长度为 2 的子串。
```
```python
class Solution:
    def hasAllCodes(self, s: str, k: int) -> bool:
        s_len = len(s)
        if s_len < 2 ** k:
            return False
        curr = set()
        for i in range(s_len+1-k):
            curr.add(s[i:k+i])
        print(curr)
        if len(curr) == 2 ** k:
            return True
        else:
            return False
```

#### [5410. 课程安排 IV](https://leetcode-cn.com/problems/course-schedule-iv/)
```python
from collections import defaultdict, deque
import functools
class Solution:
    def checkIfPrerequisite(self, n: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        connect = defaultdict(list)
        for requisite in prerequisites:
            prev, curr = requisite
            connect[prev].append(curr)

        # print(connect)
        results = []
        # @functools.lru_cache(None)

        def helper(prev, end):
            if prev == end:
                return True
            if prev not in connect:
                memo[prev] = False
                return False

            for curr in connect[prev]:
                if curr in memo:
                    ans = memo[curr]
                else:
                    ans = helper(curr, end)
                    memo[curr] = ans
                if ans == True:
                    break
            return ans

        for query in queries:
            start, end = query
            memo = {}
            ans = helper(start, end)
            results.append(ans)

        return results

        # for query in queries:
        #     start, end = query
        #     queue = deque([start])
        #     visited = set([start])
        #     flag = False
        #     while queue:
        #         prev = queue.pop()
        #         if prev == end:
        #             flag = True
        #             break
        #         if prev not in connect:
        #             continue
        #         for curr in connect[prev]:
        #             if curr not in visited:
        #                 visited.add(curr)
        #                 queue.appendleft(curr)
        #     results.append(flag)
        # return results
```

#### [5411. 摘樱桃 II](https://leetcode-cn.com/problems/cherry-pickup-ii/)
```python
import functools
class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        col_range = [-1,0,1]
        @functools.lru_cache(None)
        def dp(row, col1, col2):
            if row == n:
                return 0
            res = grid[row][col1]
            if col1 != col2:
                res += grid[row][col2]
            max_value = 0
            for delta1 in col_range:
                col1n = col1 + delta1
                if col1n < 0 or col1n >= m:
                    continue
                for delta2 in col_range:
                    col2n = col2 + delta2
                    if col2n < 0 or col2n >= m:
                        continue
                    max_value = max(max_value, dp(row+1, col1n, col2n))
            return max_value + res

        return dp(0, 0, m-1)
```
#### [114. 二叉树展开为链表](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/)
```python
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.left = left
        self.right = right
        self.val = val
class Solution:
    def flatten(self, root: TreeNode) -> None:
        def helper(root):
            if root == None:
                return
            # 将根节点的左子树变成链表
            helper(root.left)
            # 将根节点的右子树变成链表
            helper(root.right)
            temp = root.right
            # 把树的右边换成左边的链表
            root.right = root.left
            # 将左边置空
            root.left = None
            # 找到树的最右边的节点
            while root.right:
                root = root.right
            # 把右边的链表接到刚才树的最右边的节点
            root.right = temp
        helper(root)
```
#### [128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/)
```
给定一个未排序的整数数组，找出最长连续序列的长度。要求算法的时间复杂度为 O(n)。
输入: [100, 4, 200, 1, 3, 2]   输出: 4
解释: 最长连续序列是 [1, 2, 3, 4]。它的长度为 4。
```
哈希map倒序查询,巧妙O(n)
```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        lookup = set(nums)
        max_len = 0
        for num in lookup:
            if num-1 in lookup:
                continue
            curr_len = 0
            while num in lookup:
                curr_len += 1
                num += 1
            max_len = max(max_len, curr_len)
        return max_len
```
#### [837. 新21点](https://leetcode-cn.com/problems/new-21-game/)
```python
class Solution:
    def new21Game(self, N: int, K: int, W: int) -> float:
        """dp以K为界限,分为两部分"""
        dp = [0] * (K+W)
        # 当分数大于等于K,停止抽牌,此时如果<=N,获胜概率为1,否则为0
        for i in range(K, K+W):
            dp[i] = 1 if i <= N else 0
        # s为长度为W的窗口内的概率和
        s = sum(dp)
        # 当分数小于K,可以抽牌,范围是[1,W],获胜概率为窗口s内的概率和/W
        for i in range(K-1, -1, -1):
            dp[i] = s / W
            s = s - dp[i+W] + dp[i]
        return dp[0]
```
#### [15. 三数之和](https://leetcode-cn.com/problems/3sum/)
```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        results = []
        for p0 in range(n-2):
            # 如果p0已经大于0了,p1,p2必定大于0,break
            if nums[p0] > 0:
                break
            # 如果遇到重复数字,跳过
            if p0 != 0 and nums[p0] == nums[p0-1]:
                continue
            p1, p2 = p0+1, n-1
            while p1 < p2:
                if nums[p0] + nums[p1] + nums[p2] < 0:
                    p1 += 1
                elif nums[p0] + nums[p1] + nums[p2] > 0:
                    p2 -= 1
                else:
                    results.append([nums[p0],nums[p1],nums[p2]])
                    p1 += 1
                    p2 -= 1
                    # 找到三元数后,对于重复的数字跳过
                    while p1 < p2 and nums[p1] == nums[p1-1]:
                        p1 += 1
                    while p1 < p2 and nums[p2] == nums[p2+1]:
                        p2 -= 1
        return results
```
#### [18. 四数之和](https://leetcode-cn.com/problems/4sum/)
```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        """注意与三数之和的区别,target可以为负"""
        nums.sort()
        n = len(nums)
        results = []
        for p0 in range(n-3):
            # 当数组最小值和大于target break
            if nums[p0]+nums[p0+1]+nums[p0+2]+nums[p0+3] > target:
                break
            # 当数组最大值和小于target 寻找下一个数字
            if nums[p0]+nums[n-1]+nums[n-2]+nums[n-3] < target:
                continue
            # 重复数 跳过
            if p0 != 0 and nums[p0] == nums[p0-1]:
                continue
            for p1 in range(p0+1, n-2):
                # 当数组最小值和大于target break
                if nums[p0]+nums[p1]+nums[p1+1]+nums[p1+2] > target:
                    break
                # 当数组最大值和小于target 寻找下一个数字
                if nums[p0]+nums[p1]+nums[n-2]+nums[n-1] < target:
                    continue
                # 重复数 跳过
                if p1 != p0+1 and nums[p1] == nums[p1-1]:
                    continue
                p2, p3 = p1+1, n-1
                while p2 < p3:
                    val = nums[p0]+nums[p1]+nums[p2]+nums[p3]
                    if val < target:
                        p2 += 1
                    elif val > target:
                        p3 -= 1
                    else:
                        results.append([nums[p0],nums[p1],nums[p2],nums[p3]])
                        p2 += 1
                        p3 -= 1
                        while p2 < p3 and nums[p2] == nums[p2-1]:
                            p2 += 1
                        while p2 < p3 and nums[p3] == nums[p3+1]:
                            p3 -= 1
        return results
```

#### [面试题29. 顺时针打印矩阵](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)
模拟题， 收缩四个边界， 在边界范围内打印。
```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if len(matrix) == 0: return []
        l, t, r, b = 0, 0, len(matrix[0]), len(matrix)
        results = []
        while True:
            for j in range(l,r):
                results.append(matrix[t][j])
            t += 1
            if t == b: break

            for i in range(t,b):
                results.append(matrix[i][r-1])
            r -= 1
            if r == l: break

            for j in range(r-1, l-1, -1):
                results.append(matrix[b-1][j])
            b -= 1
            if t == b: break

            for i in range(b-1, t-1, -1):
                results.append(matrix[i][l])
            l += 1
            if r == l: break

        return results
```

#### [990. 等式方程的可满足性](https://leetcode-cn.com/problems/satisfiability-of-equality-equations/)
典型并查集
```python
class UnionFindSet:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0] * n

    def find(self, x):
        """返回根节点的同时完全压缩 ~= O(1)"""
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """合并两个节点到同一根节点,并维护rank O(1)"""
        px, py = self.find(x), self.find(y)
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[px] = py
            self.rank[py] += 1

    def is_connect(self, x, y):
        """查询两个节点是否联通"""
        return self.find(x) == self.find(y)

class Solution:
    def equationsPossible(self, equations: List[str]) -> bool:
        unionfind = UnionFindSet(26)
        for item in equations:
            if item[1] == "=":
                index1 = ord(item[0]) - ord("a")
                index2 = ord(item[3]) - ord("a")
                unionfind.union(index1, index2)
        for item in equations:
            if item[1] == "!":
                index1 = ord(item[0]) - ord("a")
                index2 = ord(item[3]) - ord("a")
                if unionfind.is_connect(index1, index2):
                    return False
        return True
```

#### [218. 天际线问题](https://leetcode-cn.com/problems/the-skyline-problem/)
```python
class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        # 扫描线算法
        def merge(left, right):
            p0, p1 = 0, 0
            n0, n1 = len(left), len(right)
            lh, rh = 0, 0
            merged = []
            while p0 < n0 and p1 < n1:
                # 如果横坐标左<右,记录左矩形h与之前rh的max作为当前点的h
                if left[p0][0] < right[p1][0]:
                    cp = [left[p0][0], max(rh, left[p0][1])]
                    lh = left[p0][1]
                    p0 += 1
                # 注意是elif,不然有bug
                elif left[p0][0] > right[p1][0]:
                    cp = [right[p1][0], max(lh, right[p1][1])]
                    rh = right[p1][1]
                    p1 += 1
                # 如果横坐标相等,取两点的最高高度,p0+1,p1+1
                else:
                    cp = [right[p1][0], max(left[p0][1], right[p1][1])]
                    lh = left[p0][1]
                    rh = right[p1][1]
                    p0 += 1
                    p1 += 1
                # 如果相对于上一个点,高度没有更换,不更新
                if len(merged) == 0 or cp[1] != merged[-1][1]:
                    merged.append(cp)
            merged.extend(left[p0:] or right[p1:])
            return merged

        def mergeSort(buildings):
            # return 单个矩形的左上,右下坐标
            if len(buildings) == 1:
                return [[buildings[0][0], buildings[0][2]], [buildings[0][1], 0]]
            mid = len(buildings) // 2
            left = mergeSort(buildings[mid:])
            right = mergeSort(buildings[:mid])
            return merge(left, right)

        # O(nlogn)
        if len(buildings) == 0: return []
        return mergeSort(buildings)
```

#### [面试题46. 把数字翻译成字符串](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)
动态规划,搜到了返回1,注意不用for,控制index移动.注意06只有一种可能
```python
import functools
class Solution:
    def translateNum(self, num: int) -> int:
        num = str(num)
        n = len(num)
        @functools.lru_cache(None)
        def helper(index):
            if index == n:
                return 1
            if index+2 <= n and num[index] != "0" and int(num[index:index+2]) < 26:
                return helper(index+1) + helper(index+2)
            else:
                return helper(index+1)
        return helper(0)
```

#### [238. 除自身以外数组的乘积](https://leetcode-cn.com/problems/product-of-array-except-self/)
构造L, R, 数组,存储的是该元素左边/右边的累计乘积.为了节省空间,重复利用L,并且R在正向遍历时构建更新
```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ans = [0] * n
        ans[-1] = nums[-1]
        for i in range(n-2,-1,-1):
            ans[i] = ans[i+1] * nums[i]
        R = 1
        for i in range(n):
            if i < n-1:
                ans[i] = R * ans[i+1]
            else:
                ans[i] = R
            R *= nums[i]
        return ans
```

#### [26. 删除排序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)
```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        l = 0
        for r in range(len(nums)):
            if nums[r] != nums[l]:
                l += 1
                nums[l] = nums[r]
        return l+1

        # l, r = 0, 0
        # n = len(nums)
        # while r < len(nums):
        #     val = nums[r]
        #     r += 1
        #     while r < n and nums[r] == val:
        #         r += 1
        #     nums[l] = val
        #     l += 1
        # return l
```

#### [27. 移除元素](https://leetcode-cn.com/problems/remove-element/)
```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        l = 0
        for r in range(len(nums)):
            if nums[r] != val:
                nums[l] = nums[r]
                l += 1
        return l
```

#### [576.出界的路径数](https://leetcode-cn.com/problems/out-of-boundary-paths/)
动态规划,搜索
```python
import functools
class Solution:
    def findPaths(self, m: int, n: int, N: int, i: int, j: int) -> int:
        oriens = [(-1,0),(1,0),(0,-1),(0,1)]
        @functools.lru_cache(None)
        def helper(N, i, j):
            if i < 0 or i >= m or j < 0 or j >= n:
                return 1
            if N == 0:
                return 0
            res = 0
            for orien in oriens:
                ni, nj = i+orien[0], j+orien[1]
                res += helper(N-1, ni, nj)
            return res
        return helper(N, i, j) % (10**9+7)
```

#### [1014. 最佳观光组合](https://leetcode-cn.com/problems/best-sightseeing-pair/)
维护 mx = max(A[i]+i), ans = max(ans, mx + (A[i]-i))
```python
class Solution:
    def maxScoreSightseeingPair(self, A: List[int]) -> int:
        mx = A[0] + 0
        n = len(A)
        ans = 0
        for i in range(1, n):
            ans = max(ans, mx + (A[i]-i))
            mx = max(mx, A[i]+i)
        return ans
```

#### [71. 简化路径](https://leetcode-cn.com/problems/simplify-path/)
```python
class Solution:
    def simplifyPath(self, path: str) -> str:
        """
        1. 用 / 分割path
        2. 遍历分割后的path,遇到..则stack.pop(),遇到合法路径append
        """
        path = path.split("/")
        stack = []
        for item in path:
            if item == "..":
                if stack:
                    stack.pop()
            elif item and item != ".":
                stack.append(item)
        clean_path = "/" + "/".join(stack)
        return clean_path
```

#### [93. 复原IP地址](https://leetcode-cn.com/problems/restore-ip-addresses/)
核心思路是用回溯,找出所有合法的ip组合.
ip节的合法长度是1-3,因此回溯的主结构是 for i in range(3)
当前的边界=上一个ip节的index+当前ip节长度+1. curr = index+i+1
然后剪掉非法ip的情况,最后index如果走到最后并且有4个ip节,保存结果

```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        n = len(s)
        result = []
        def helper(index, cnt, res):
            if index == n and cnt == 4:
                result.append(res[1:])
            # 每个ip 长度 1-3
            for i in range(3):
                curr = index+i+1
                rest = n - curr
                # 剩余个数无法凑出合法ip, 剪枝
                if rest > (4-cnt-1) * 3:
                    continue
                # 超过ip最大长度, 剪枝
                if curr > n:
                    break
                ip = s[index:curr]
                # 首尾为0,非法ip
                if len(ip) > 1 and ip[0] == "0":
                    continue
                # 大于255, 非法ip
                if int(ip) > 255:
                    continue
                helper(curr, cnt+1, res+"."+ip)
        helper(0, 0, "")
        return result
```
#### [二叉树的锯齿形层次遍历](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)
双栈stack(left,right), stack_inv(right,left)
```python
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if root == None: return []
        stack, stack_inv = [root], []
        result = []
        while True:
            line = []
            while stack:
                top = stack.pop()
                line.append(top.val)
                if top.left:
                    stack_inv.append(top.left)
                if top.right:
                    stack_inv.append(top.right)
            if line:
                result.append(line)
            else:
                break
            line = []
            while stack_inv:
                top = stack_inv.pop()
                line.append(top.val)
                if top.right:
                    stack.append(top.right)
                if top.left:
                    stack.append(top.left)
            if line:
                result.append(line)
            else:
                break
        return result
```

#### [240. 搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)
```python
class Solution:
    def searchMatrix(self, matrix, target):
        """正是因为严格的升序,所以可以用区间排除"""
        n = len(matrix)
        if n == 0: return False
        m = len(matrix[0])
        row, col = n-1, 0
        while row >= 0 and col < m:
            if matrix[row][col] > target:
                row -= 1
            elif matrix[row][col] < target:
                col += 1
            else:
                return True
        return False
```

#### [1296. 划分数组为连续数字的集合](https://leetcode-cn.com/problems/divide-array-in-sets-of-k-consecutive-numbers/)
同一手顺子, 模拟题. 注意每次-freq.
```python
from collections import Counter
class Solution:
    def isPossibleDivide(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        if n % k != 0: return False
        stat = Counter(nums)
        for key in sorted(stat):
            freq = stat[key]
            if freq > 0:
                for item in range(key, key+k):
                    if item in stat and stat[item] >= freq:
                        stat[item] -= freq
                    else:
                        return False
        return True
```

#### [面试题 16.18. 模式匹配](https://leetcode-cn.com/problems/pattern-matching-lcci/)
找出所有可行的(la,lb)组，然后进行组合测试。
```python
class Solution:
    def patternMatching(self, pattern: str, value: str) -> bool:
        if not pattern: return not value
        if not value: return len(pattern)<=1
        # 1、清点字符
        ca = pattern.count('a')
        cb = len(pattern) - ca
        # 2、只有一种字符
        if 0==ca*cb:
            return value==value[:len(value)//len(pattern)]*len(pattern)                
        # 3、如果有两种字符
        for la in range(len(value)//ca+1):
            # len(value) == la*ca + lb*cb
            if 0 != (len(value)-la*ca)%cb: continue
            p,lb = 0,(len(value)-la*ca)//cb
            a,b = set(),set()
            # 分离子串
            for c in pattern:
                if c=='a':
                    a.add(value[p:p+la])
                    p += la
                else:
                    b.add(value[p:p+lb])
                    p += lb
            if len(a)==len(b)==1: return True
        return False
```

#### [343. 整数拆分](https://leetcode-cn.com/problems/integer-break/)
```python
import functools
class Solution:
    def integerBreak(self, n: int) -> int:
        @functools.lru_cache(None)
        def helper(index):
            if index == 1:
                return 1
            res = 0
            for i in range(1, index):
                split = i * helper(index-i)
                not_split = i * (index-i)
                res = max(res, split, not_split)
            return res
        return helper(n)
```

#### [4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)
```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        def helper(nums1, nums2, k):
            if len(nums1) < len(nums2):
                return helper(nums2, nums1, k)
            if len(nums2) == 0:
                return nums1[k-1]
            if k == 1:
                return min(nums1[0], nums2[0])

            t = min(k//2, len(nums2))
            if nums1[t-1] < nums2[t-1]:
                return helper(nums1[t:], nums2, k-t)
            else:
                return helper(nums1, nums2[t:], k-t)

        k1 = (len(nums1) + len(nums2) + 1) // 2
        k2 = (len(nums1) + len(nums2) + 2) // 2
        if k1 == k2:
            return helper(nums1, nums2, k1)
        else:
            return (helper(nums1, nums2, k1) + helper(nums1, nums2, k2)) / 2
```

#### [剑指 Offer 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)
```python
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        n = len(nums)
        arr = [(i, nums[i]) for i in range(n)]
        self.res = 0

        def merge(arr_l, arr_r):
            arr = []
            n1, n2 = len(arr_l), len(arr_r)
            p1, p2 = 0, 0
            while p1 < n1 or p2 < n2:
                # 注意是 <=
                if p2 == n2 or (p1 < n1 and arr_l[p1][1] <= arr_r[p2][1]):
                    self.res += p2
                    arr.append(arr_l[p1])
                    p1 += 1
                else:
                    arr.append(arr_r[p2])
                    p2 += 1
            return arr

        def mergeSort(arr, l, r):
            if r == 0:
                return
            if l == r - 1:
                return [arr[l]]
            m = l + (r-l) // 2
            arr_l = mergeSort(arr, l, m)
            arr_r = mergeSort(arr, m, r)
            return merge(arr_l, arr_r)

        mergeSort(arr, 0, n)
        return self.res
```
