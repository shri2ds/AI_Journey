class Node:
  """ Node for Binary Tree. """
  def __init__(self, val=0, right=None, left=None):
    self.val = val
    self.left = left
    self.right = right

def LowestCommonAncestor(root:Node, p:Node, q:Node) -> Node :
    """
        Return the Lowest Common Ancestor from the BST given the root and two target nodes.

        Args:
            root: Root node of the Binary Search Tree.
            p: First target node whose ancestor we seek.
            q: Second target node whose ancestor we seek.
        
        Returns: 
            Node: The lowest common ancestor node if both targets exist in the tree, else None.
    """
    node = root

    while node:
        if p.val > node.val and q.val > node.val:
            node = node.right
        elif p.val < node.val and q.val < node.val:
            node = node.left
        else:
            return node


if __name__ == "__main__":
    # Sample example
    
    # Build BST from [6,2,8,0,4,7,9,None,None,3,5]
    node0 = Node(0)
    node3 = Node(3)
    node5 = Node(5)
    node4 = Node(4, node3, node5)
    node2 = Node(2, node0, node4)
    node7 = Node(7)
    node9 = Node(9)
    node8 = Node(8, node7, node9)
    root = Node(6, node2, node8)
    
    p = node2  
    q = node8  
    ancestor = LowestCommonAncestor(root, p, q)
    print(ancestor.val if ancestor else None)
