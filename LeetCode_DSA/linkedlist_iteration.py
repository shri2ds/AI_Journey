class ListNode:
    """
    Definition for singly-linked list node.
    """
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def create_linked_list(values: list) -> ListNode:
    """
    Create a singly linked list from a list of values.

    Args:
        values (list): List of elements (int, str, tuple, etc.) to convert into linked list nodes.

    Returns:
        ListNode: Head node of the constructed linked list.
    """
    if not values:
        return None

    head = ListNode(values[0])
    current = head
    for value in values[1:]:
        current.next = ListNode(value)
        current = current.next

    return head


def reverse_linked_list_iterative(head: ListNode) -> ListNode:
    """
    Reverse a singly linked list using iterative approach.

    Args:
        head (ListNode): Head node of the linked list.

    Returns:
        ListNode: Head of the reversed linked list.
    """
    previous = None
    current = head

    while current:
        next_node = current.next      # Save next node
        current.next = previous       # Reverse the link
        previous = current            # Move previous to current
        current = next_node           # Move to next node

    return previous


if __name__ == "__main__":
    # Sample example
    head = create_linked_list([1, 2, 3])
    reversed_head = reverse_linked_list_iterative(head)
    print("Reversed Linked List:")
    print(reversed_head)
