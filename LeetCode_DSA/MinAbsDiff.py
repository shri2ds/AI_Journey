def min_abs_diff(arr):
    """
    Find minimum absolute difference between sum of two partitions.
    
    Split array at position P (1 <= P < N) into:
      - Left:  arr[0] to arr[P-1]
      - Right: arr[P] to arr[N-1]
    
    Time: O(n), Space: O(1)
    """
    n = len(arr)
    total_sum = sum(arr)
    left_sum = 0
    min_diff = float('inf')
    
    for i in range(1, n):  # P from 1 to N-1 (both sides must have elements)
        left_sum += arr[i - 1]
        right_sum = total_sum - left_sum
        diff = abs(left_sum - right_sum)
        min_diff = min(min_diff, diff)
    
    return min_diff


if __name__ == "__main__":
    test_cases = [
        ([3, 1, 2, 4, 3], 1),    # P=3: |6-7|=1
        ([-1, -2, -3, -4], 2),   # P=3: |-6-(-4)|=2
        ([1, 2], 1),             # P=1: |1-2|=1
        ([1, 2, 3], 0),          # P=2: |3-3|=0
        ([10, 20, 30], 10),      # P=2: |30-30|=0? No, P=1: |10-50|=40, P=2: |30-30|=0
        ([5, 5, 5, 5], 0),       # P=2: |10-10|=0
    ]
    
    print("="*50)
    print("MIN ABSOLUTE DIFFERENCE TEST CASES")
    print("="*50)
    
    for arr, expected in test_cases:
        result = min_abs_diff(arr)
        status = "✓" if result == expected else "✗"
        print(f"{status} Input: {arr}")
        print(f"  Expected: {expected}, Got: {result}")
        print()
