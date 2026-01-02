class Node:
    """Node for a binary tree."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def diameterOfBinaryTree(root):
    """ Returns the diameter of the binary tree """
    self.diameter = 0 # Initialising the diameter with 0

    # DFS function here returns height of the binary tree
    def dfs(current):
        if not current:
            return 0
        
        left = dfs(current.left)
        right = dfs(current.right)

        self.result = max(self.diameter, left + right) # Updating the diameter accordingly
        return 1 + max(left, right)
        
      dfs(root)
      return self.diameter

if __name__ == "__main__":
    # Example usage
    root = Node(3, Node(9, Node(13)), Node(20, Node(15), Node(7)))
    result = diameterOfBinaryTree(root)
    print(result)
