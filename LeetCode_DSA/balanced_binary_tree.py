class Node:
    """Node for a binary tree."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def isBalanced(root):
    # function that returns height or -1
    def height(node):
        if not node:
            return 0

        # Check left & right
        left_h = height(node.left)
        if left_h == -1:
            return -1
        right_h = height(node.right)
        if right_h == -1:
            return -1

        # Check Balance
        if abs(left_h - right_h) > 1:
            return -1

        # Return my height
        return 1 + max(left_h, right_h)

    # If the function returns -1, it's False. Otherwise True.
    return height(root) != -1

if __name__ == "__main__":
    # Example usage
    root = Node(3, Node(9, Node(13)), Node(20, Node(15), Node(7)))
    result = isBalanced(root)
    print(result)

