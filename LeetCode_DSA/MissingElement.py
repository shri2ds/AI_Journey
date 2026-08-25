def MissingElement(elements):
    """
    Find the missing element in an array containing elements from 1 to N+1.
    
    Time: O(n), Space: O(1)
    
    Logic: expected_sum - actual_sum = missing element
    """
    n = len(arr) + 1
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(arr)
    return expected_sum - actual_sum

if __name__ == "__main__":
    test_cases = [
        ([2, 3, 1, 5], 4),       # 4 is missing from 1..5
        ([1], 2),                # only 1 present, so 2 is missing
        ([2], 1),                # only 2 present, so 1 is missing
        ([], 1),                 # empty array, 1 is missing
        ([1, 2, 3, 4, 5], 6),    # nothing missing, so N+1 = 6
        ([1, 3], 2),             # 2 is missing from 1..3
    ]
    
    print("="*50)
    print("MISSING ELEMENT TEST CASES")
    print("="*50)
    
    for arr, expected in test_cases:
        result = find_missing_element(arr)
        status = "✓" if result == expected else "✗"
        print(f"{status} Input: {arr}")
        print(f"  Expected: {expected}, Got: {result}")
        print()
