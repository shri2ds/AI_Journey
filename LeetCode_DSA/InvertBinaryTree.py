from collections import deque

class Node:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def invertTree(root):
    """ Invert a binary tree """
    if not root:
        return None

    root.left, root.right = root.right, root.left

    invertTree(root.left)
    invertTree(root.right)

    return root

def printTree(root):
    """Return a level-order list representation of the tree"""
    if not root:
        return None

    result = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        if node:
            result.append(node.val)
            result.append(node.left)
            result.append(node.right)
        else:
            result.append(None)

    while result and result[-1] is None:
        result.pop()
    
    return result

if __name__ = "__main__":
    # Sample Example
    root = Node(3, Node(9, Node(13)), Node(20, Node(15), Node(7)))
    print("Original tree:", printTree(root))
    inverted = invertTree(root)
    print("Inverted tree:", printTree(inverted))
