def find_leader_hashmap(arr):
    """
    Find index of leader element using HashMap.
    
    Simple approach: Count occurrences, then find element with count > n/2.
    
    Time: O(n), Space: O(n)
    """
    if not arr:
        return -1
    
    counts = {}
    for num in arr:
        counts[num] = counts.get(num, 0) + 1
    
    for key, value in counts.items():
        if value > len(arr) // 2:
            return arr.index(key)
    
    return -1


# ============================================================
# APPROACH 2: Boyer-Moore Voting Algorithm (Space Optimized)
# Time: O(n), Space: O(1)
# ============================================================

def find_leader_boyer_moore(arr):
    """
    Find index of leader element using Boyer-Moore Voting.
    
    Phase 1: Find candidate (element that survives voting)
    Phase 2: Verify candidate actually appears > n/2 times
    
    Time: O(n), Space: O(1)
    """
    if not arr:
        return -1
    
    # Phase 1: Find candidate
    candidate = None
    count = 0
    candidate_index = 0
    
    for i, num in enumerate(arr):
        if count == 0:
            candidate = num
            candidate_index = i
            count = 1
        elif num == candidate:
            count += 1
        else:
            count -= 1
    
    # Phase 2: Verify candidate is actually a leader (> n/2)
    actual_count = arr.count(candidate)
    
    if actual_count > len(arr) // 2:
        return candidate_index
    
    return -1


if __name__ == "__main__":
    test_cases = [
        ([4, 3, 4, 4, 4, 2], 0),   # 4 appears 4/6 times
        ([1, 2, 3, 4, 5], -1),     # No leader
        ([1, 1, 2], 0),            # 1 appears 2/3 times
        ([5], 0),                  # Single element is leader
        ([1, 1, 2, 2], -1),        # Tie - exactly half each
        ([], -1),                  # Empty array
        ([2, 2, 2, 3, 3], 0),      # 2 appears 3/5 times
        ([1, 2, 1, 2, 1], 0),      # 1 appears 3/5 times
    ]
    
    print("="*60)
    print("LEADER ELEMENT - APPROACH 1: HashMap")
    print("="*60)
    
    for arr, expected in test_cases:
        result = find_leader_hashmap(arr)
        if expected == -1:
            status = "✓" if result == -1 else "✗"
        else:
            leader_val = arr[expected] if arr else None
            status = "✓" if (result != -1 and arr[result] == leader_val) else "✗"
        print(f"{status} Input: {str(arr):25} Got: {result}")
    
    print("\n" + "="*60)
    print("LEADER ELEMENT - APPROACH 2: Boyer-Moore")
    print("="*60)
    
    for arr, expected in test_cases:
        result = find_leader_boyer_moore(arr)
        if expected == -1:
            status = "✓" if result == -1 else "✗"
        else:
            leader_val = arr[expected] if arr else None
            status = "✓" if (result != -1 and arr[result] == leader_val) else "✗"
        print(f"{status} Input: {str(arr):25} Got: {result}")
    
