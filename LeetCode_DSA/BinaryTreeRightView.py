from collections import deque

class Node:
    """Simple binary tree node."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def rightSideView(root):
    """Return the nodes visible when looking at the tree from the right side."""
    queue = deque([root])
    right_view = []
    if root:
        right_view.append(root.val)

        while queue:
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            layer = [i.val for i in queue]  # collect current level's nodes
            if layer:
                right_view.append(layer[-1])  # last value is the visible right node

    return right_view

if __name__ == "__main__":
    # Example usage
    # root = None # Tested with edge case
    root = Node(3, Node(9, Node(13)), Node(20, Node(15), Node(7)))
    result = rightSideView(root)
    print(result)
