# 面试题目整理

### [面试题07. 重建二叉树](https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/)
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        n = len(preorder)
        # 建立哈希表，实现O(1)查询
        lookup_table = {inorder[i]: i for i in range(n)}
        # 递归中维护子树根index与子树区间范围(相对于preorder)
        def helper(root_i, left, right):
            # 如果区间相交，return叶子节点的None
            if left >= right: return
            root = TreeNode(preorder[root_i])
            # 查询子树根在中序遍历中的位置
            in_i = lookup_table[preorder[root_i]]
            # 左子树root index 根+1
            root.left = helper(root_i+1, left, in_i)
            # 右子树root index 根+左子树长度+1
            root.right = helper(root_i+in_i-left+1, in_i+1, right)
            # 层层向上返回子树的根
            return root

        root = helper(0, 0, n)
        return root
```

### [108. 将有序数组转换为二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/)
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        def helper(left, right):
            if left >= right: return None
            mid = left + (right-left)//2
            root = TreeNode(nums[mid])
            root.left = helper(left, mid)
            root.right = helper(mid+1, right)
            return root
        return helper(0, len(nums))
```
