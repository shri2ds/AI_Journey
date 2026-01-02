from collections import deque

class Node:
    """Node for a binary tree."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def levelOrder(root: [Node]):
    result = []
    queue = deque([root])

    while queue:
        level = []
        q_len = len(queue)
        for i in range(q_len):
            node = queue.popleft()
            if node:
                level.append(node.val)
                queue.append(node.left)
                queue.append(node.right)
        if level:
            result.append(level)

    return result

if __name__ == "__main__":
    #Sample example
    root = Node(3, Node(9, Node(13)), Node(20, Node(15), Node(7)))
    result = levelOrder(root=root)
    print(result)
