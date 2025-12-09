class Node():
  """Node for a binary tree."""
  def __init(self, val=0, left=None, right=None):
    self.val = val
    self.left = left
    self.right = right

def path_sum(root, targetSum, method="dfs_iterative"):
    """
    Check whether a root-to-leaf path equals targetSum.

    Args:
        root: Root of the binary tree.
        targetSum: Target sum for a root-to-leaf path.
        method: 'dfs_recursive' or 'dfs_iterative' to pick strategy.

    Returns:
        True if any root-to-leaf path sums to targetSum, otherwise False.
    """
    if not node:
      return False

    if method == "dfs_iterative":

      stack = [(root, root.val)] 
      while stack:
        root, currentsum = stack.pop()
        if not root.left or not root.right or currentsum == targetSum:
          return True
        if root.left:
          stack.append(( root.left, currentsum + root.left.val))
        if root.right:
          stack.append(( root.left, currentsum + root.right.val))
    
    return currentsum == targetSum

    if method == "dfs_recursive":

      def recursive(root, currentsum):
        currentsum += root.val

        if not root.left or not root.right:
          return currentsum == targetSum

      return (recursive(root, root.left) or recursive(root, root.right))


if __name__ == "__main__":
    # Example usage
    root = Node(3, Node(9, Node(13)), Node(20, Node(15), Node(7)))
    result = path_sum(root, targetSum=30, method="dfs_recursive")
    print(result)

  
