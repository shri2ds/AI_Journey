class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeTwoLists(self, list1: ListNode, list2: ListNode) -> ListNode:
        """
        Merge two sorted linked lists into one sorted list.
        """
        result = ListNode()
        current = result
        while list1 and list2:
            if list1.val < list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next
            current = current.next
        if list1:
            current.next = list1
        if list2:
            current.next = list2
        return result.next

# A sample example
if __name__ == "__main__":
    # Helper to print
    def print_list(node):
        vals = []
        while node:
            vals.append(str(node.val))
            node = node.next
        print(" -> ".join(vals))


    # List 1: 1 -> 2 -> 4
    l1 = ListNode(1, ListNode(2, ListNode(4)))
    # List 2: 1 -> 3 -> 4
    l2 = ListNode(1, ListNode(3, ListNode(4)))

    sol = Solution()
    merged = sol.mergeTwoLists(l1, l2)
    print("Merged List:")
    print_list(merged)
