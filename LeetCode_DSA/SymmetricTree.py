class Node:
    """Simple binary tree node."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def isSymmetric(self, root: Node) -> bool:
        if not root:
            return True

        def isMirror(t1, t2):

            if not t1 and not t2: return True
            if not t1 or not t2: return False

            return (t1.val == t2.val) and isMirror(t1.left, t2.right) and isMirror(t1.right, t2.left)

        return isMirror(root.left, root.right)


if __name__ == "__main__":
    # Sample example
    node3_L = Node(3)
    node4_L = Node(4)
    node3_R = Node(3)
    node4_R = Node(4)

    left_child = Node(2, node3_L, node4_L)
    right_child = Node(2, node4_R, node3_R)
    root_symmetric = Node(1, left_child, right_child)

    Sol = Solution()

    print(f"Test Case (Expected True): {Sol.isSymmetric(root_symmetric)}")
