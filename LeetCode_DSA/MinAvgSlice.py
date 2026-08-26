def min_avg_slice(arr):
    """
    Find the starting index of the slice with minimum average.
    
    The minimum average slice will always have length 2 or 3. Any longer slice can be broken into smaller slices, and at least one of them will have average <= the original slice's average.
    
    Time: O(n), Space: O(1)
    
    Args:
        arr: List of integers (length >= 2)
    
    Returns:
        Starting index of the slice with minimum average
    """
    n = len(arr)
    min_avg = float('inf')
    min_index = 0
    
    for i in range(n - 1):
        # Check slice of length 2
        avg_2 = (arr[i] + arr[i + 1]) / 2
        if avg_2 < min_avg:
            min_avg = avg_2
            min_index = i
        
        # Check slice of length 3
        if i < n - 2:
            avg_3 = (arr[i] + arr[i + 1] + arr[i + 2]) / 3
            if avg_3 < min_avg:
                min_avg = avg_3
                min_index = i
    
    return min_index


if __name__ == "__main__":
    test_cases = [
        ([4, 2, 2, 5, 1, 5, 8], 1),   # slice [2,2] at index 1, avg=2.0
        ([1, 2], 0),                  # only one slice possible
        ([-5, 4, 3], 0),              # slice [-5,4] at index 0, avg=-0.5
        ([-5, 4, 3, -9], 2),          # slice [3,-9] at index 2, avg=-3.0
        ([10, 10, -1, 2, 4, -1, 2, -1], 5),  # slice [-1,2,-1] at index 5
    ]
    
    print("="*50)
    print("MIN AVERAGE SLICE TEST CASES")
    print("="*50)
    
    for arr, expected in test_cases:
        result = min_avg_slice(arr)
        status = "✓" if result == expected else "✗"
        print(f"{status} Input: {arr}")
        print(f"  Expected: {expected}, Got: {result}")
        print()
