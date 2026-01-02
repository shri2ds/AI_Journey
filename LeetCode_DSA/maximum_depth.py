from collections import deque

class Node:
    """Node for a binary tree."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def maxDepth(root: Node, method: str = "bfs"):
    """Compute the maximum depth of a binary tree using the chosen traversal.

    Args:
        root: The root node of the binary tree.
        method: One of "dfs_recursive", "dfs_iterative", or "bfs" to select the traversal strategy.

    Returns:
        The maximum depth as an integer.

    Raises:
        ValueError: If an unsupported method is provided.
    """

    # Base Case: Empty node contributes 0 depth
    if not root:
        return 0

    if method == "dfs_recursive":
        return 1 + max(
            maxDepth(root.left, method),
            maxDepth(root.right, method)
        )

    if method == "dfs_iterative":
        stack = [(root, 1)]
        result = 0

        while stack:
            node, depth = stack.pop()
            if node:
                result = max(result, depth)
                stack.append((node.left, depth + 1))
                stack.append((node.right, depth + 1))

        return result

    if method == "bfs":
        level = 0
        queue = deque([root])
        while queue:
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            level += 1
        return level

    raise ValueError(f"Unknown method '{method}' for maxDepth")

# Simple Tree:   1
#               / \
#              2   3
# root = Node(1, Node(2), Node(3))

if __name__ == "__main__":
    # Example usage
    root = Node(3, Node(9, Node(13)), Node(20, Node(15), Node(7)))
    result = maxDepth(root, "dfs_recursive")
    print(result)
