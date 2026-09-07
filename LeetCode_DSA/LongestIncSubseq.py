from typing import List
import bisect


def length_of_lis_dp(nums: List[int]) -> int:
    """
    Find length of Longest Increasing Subsequence using Dynamic Programming.
    
    dp[i] = length of LIS ending at index i
    For each i, check all previous j where nums[j] < nums[i]
    
    Time: O(n^2), Space: O(n)
    """
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)


def length_of_lis_binary_search(nums: List[int]) -> int:
    """
    Find length of Longest Increasing Subsequence using Binary Search.
    
    Maintain 'tails' array where tails[k] = smallest tail of all 
    increasing subsequences of length k+1.
    
    For each num:
    - If num > all tails: append (extends longest subsequence)
    - Else: replace first tail >= num (keeps smaller tail for future)
    
    Time: O(n log n), Space: O(n)
    """
    if not nums:
        return 0
    
    tails = []
    
    for num in nums:
        idx = bisect.bisect_left(tails, num)
        if idx == len(tails):
            tails.append(num)
        else:
            tails[idx] = num
    
    return len(tails)


if __name__ == "__main__":
    test_cases = [
        ([10, 9, 2, 5, 3, 7, 101, 18], 4),  # [2,3,7,101] or [2,5,7,18]
        ([0, 1, 0, 3, 2, 3], 4),             # [0,1,2,3]
        ([7, 7, 7, 7, 7], 1),                # All same - strictly increasing
        ([5, 4, 3, 2, 1], 1),                # Decreasing
        ([1, 2, 3, 4, 5], 5),                # Already increasing
        ([42], 1),                           # Single element
        ([-2, -1, 0, 3, -5, 4], 5),          # Negatives: [-2,-1,0,3,4]
    ]
    
    print("="*60)
    print("LONGEST INCREASING SUBSEQUENCE")
    print("="*60)
    
    print("\n--- Approach 1: DP O(n^2) ---")
    for nums, expected in test_cases:
        result = length_of_lis_dp(nums)
        status = "✓" if result == expected else "✗"
        print(f"{status} nums={nums}")
        print(f"  Expected: {expected}, Got: {result}")
    
    print("\n--- Approach 2: Binary Search O(n log n) ---")
    for nums, expected in test_cases:
        result = length_of_lis_binary_search(nums)
        status = "✓" if result == expected else "✗"
        print(f"{status} nums={nums}")
        print(f"  Expected: {expected}, Got: {result}")
