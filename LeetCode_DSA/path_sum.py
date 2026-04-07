from collections import deque

class Node:
    """Node for a binary tree."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def hasPathSum(root: Node, targetSum: int, method: str = "dfs_iterative"):
    """Check whether a root-to-leaf path equals targetSum.

    Args:
        root: Root of the binary tree.
        targetSum: Target sum for a root-to-leaf path.
        method: 'dfs_recursive' or 'dfs_iterative' to pick strategy.

    Returns:
        True if any root-to-leaf path sums to targetSum, otherwise False.
    """

    if not root:
        return False

    if method == "dfs_recursive":

        def dfs_recursive(node, curSum):
            curSum += node.val

            if not node.left or not node.right:
                return curSum == targetSum

            return (dfs_recursive(node.left, curSum) or dfs_recursive(node.right, curSum))

        return dfs_recursive(root, 0)

    if method == "dfs_iterative":
        stack = [(root, root.val)]
        while stack:
            node, current_sum = stack.pop()
            if not node.left and not node.right and current_sum == targetSum:
                return True
            if node.right:
                stack.append((node.right, current_sum + node.right.val))
            if node.left:
                stack.append((node.left, current_sum + node.left.val))

        return False

def build_tree_from_list(values):
    """Build a binary tree from level-order list (use None for missing nodes)."""

    if not values:
        return None

    nodes = [Node(val) if val is not None else None for val in values]
    kids = nodes[1:]
    for node in nodes:
        if node and kids:
            node.left = kids.pop(0)
        if node and kids:
            node.right = kids.pop(0)
    return nodes[0]

if __name__ == "__main__":
    # Sample usage
    tree_values = [5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1]    # Build tree from [5,4,8,11,None,13,4,7,2,None,None,None,1]
    root = build_tree_from_list(tree_values)
    result = hasPathSum(root, targetSum=22, method="dfs_iterative")
    print(result)
