class Node:
    """Node for a binary tree."""
    def __init__(self, val=0, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random

class CopyListRandomPointer:
    def copyRandomList(self, head: 'Node') -> 'Node':
        OldToCopy = {None: None}
        
        # First pass: create copies of all nodes
        cur = head
        while cur:
            copy = Node(cur.val)
            OldToCopy[cur] = copy
            cur = cur.next
        
        # Second pass: assign next and random pointers
        cur = head
        while cur:
            copy = OldToCopy[cur]
            copy.next = OldToCopy[cur.next]
            copy.random = OldToCopy[cur.random]
            cur = cur.next

        return OldToCopy[head]

def build_list(arr):
    """Build linked list from array format [[val, random_index], ...]"""
    if not arr:
        return None
    
    nodes = [Node(item[0]) for item in arr]
    
    for i, node in enumerate(nodes):
        node.next = nodes[i + 1] if i + 1 < len(nodes) else None
        random_idx = arr[i][1]
        node.random = nodes[random_idx] if random_idx is not None else None
    
    return nodes[0]

def list_to_array(head):
    """Convert linked list to array format [[val, random_index], ...]"""
    if not head:
        return []
    
    nodes = []
    cur = head
    while cur:
        nodes.append(cur)
        cur = cur.next
    
    node_to_index = {node: i for i, node in enumerate(nodes)}
    
    result = []
    for node in nodes:
        random_idx = node_to_index[node.random] if node.random else None
        result.append([node.val, random_idx])
    
    return result

def format_output(arr):
    """Format array to match LeetCode style output"""
    formatted = []
    for item in arr:
        val, rand = item
        rand_str = "null" if rand is None else str(rand)
        formatted.append(f"[{val},{rand_str}]")
    return "[" + ",".join(formatted) + "]"

if __name__ == "__main__":
    solution = CopyListRandomPointer()
    
    # Example 1: [[7,null],[13,0],[11,4],[10,2],[1,0]]
    print("Example 1:")
    input1 = [[7, None], [13, 0], [11, 4], [10, 2], [1, 0]]
    head1 = build_list(input1)
    print(f"Input:  {format_output(input1)}")
    
    result1 = solution.copyRandomList(head1)
    output1 = list_to_array(result1)
    print(f"Output: {format_output(output1)}")
    
    # Example 2: [[1,1],[2,1]]
    print("\nExample 2:")
    input2 = [[1, 1], [2, 1]]
    head2 = build_list(input2)
    print(f"Input:  {format_output(input2)}")
    
    result2 = solution.copyRandomList(head2)
    output2 = list_to_array(result2)
    print(f"Output: {format_output(output2)}")
