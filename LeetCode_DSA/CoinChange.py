def coin_change(coins, target):
    """
    Find minimum number of coins needed to make target amount.
    
    Uses Dynamic Programming (bottom-up):
    - dp[i] = minimum coins needed to make amount i
    - For each amount, try all coins and pick minimum
    
    Why not Greedy? Example: coins=[1,3,4], target=6
    - Greedy: 4+1+1 = 3 coins ❌
    - Optimal: 3+3 = 2 coins ✓
    
    Time: O(target * len(coins))
    Space: O(target)
    """
    if target == 0:
        return 0
    
    # dp[i] = min coins to make amount i
    dp = [float('inf')] * (target + 1)
    dp[0] = 0  
    
    for amount in range(1, target + 1):
        for coin in coins:
            if coin <= amount and dp[amount - coin] != float('inf'):
                dp[amount] = min(dp[amount], dp[amount - coin] + 1)
    
    return dp[target] if dp[target] != float('inf') else -1


if __name__ == "__main__":
    test_cases = [
        ([1, 2, 5], 11, 3),    # 5+5+1
        ([2], 3, -1),          # Impossible (only even coins)
        ([1], 0, 0),           # Target 0 needs 0 coins
        ([1, 3, 4], 6, 2),     # 3+3 (greedy would fail here)
        ([5], 3, -1),          # Smallest coin > target
        ([1], 1, 1),           # Single coin exact match
        ([1, 5, 10], 18, 5),   # 10+5+1+1+1
    ]
    
    print("="*50)
    print("MIN COINS (Dynamic Programming)")
    print("="*50)
    
    for coins, target, expected in test_cases:
        result = coin_change(coins, target)
        status = "✓" if result == expected else "✗"
        print(f"{status} coins={coins}, target={target}")
        print(f"  Expected: {expected}, Got: {result}")
