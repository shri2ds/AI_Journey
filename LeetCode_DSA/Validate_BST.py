from collections import deque

class Node:
    """Node for a binary tree."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def isValidBST(self, root: [Node]) -> bool:
  #Iterative approach
    stack = [(root, -math.inf, math.inf)]
    while stack:
        node, low, high = stack.pop()
        if not node:
            continue
        if not (low < node.val < high):
            return False
        stack.append((node.right, node.val, high))
        stack.append((node.left, low, node.val))
    return True

  # Recursive approach
  # def isValid(root, minimum, maximum):
  #     if not root:
  #         return True 
  #     if root.val <= minimum or root.val >= maximum:
  #         return False
  #     return isValid(root.left, minimum , root.val) and isValid(root.right, root.val, maximum )
  # return isValid(root, -inf, +inf)


if __name__ == "__main__":
    #Sample example
    root = Node(2, Node(1), Node(3))
    result = isValidBST(root=root)
    print(result)
