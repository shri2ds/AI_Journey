from typing import List


def word_break(s: str, word_dict: List[str]) -> bool:
    """
    Determine if string s can be segmented into words from word_dict.
    
    Uses Dynamic Programming (bottom-up, right to left):
    - dp[i] = True if s[i:] can be segmented using words from dictionary
    - For each position, try all words and check if remaining string is valid
    
    Time: O(n * m * k) where n=len(s), m=len(word_dict), k=avg word length
    Space: O(n)
    
    Args:
        s: Input string to segment
        word_dict: List of valid dictionary words
    
    Returns:
        True if s can be segmented, False otherwise
    """
    n = len(s)
    dp = [False] * (n + 1)
    dp[n] = True  # Empty string is valid
    
    for i in range(n - 1, -1, -1):
        for word in word_dict:
            # Check if word matches at position i and remaining is valid
            if i + len(word) <= n and s[i:i + len(word)] == word:
                dp[i] = dp[i + len(word)]
            if dp[i]:
                break
    
    return dp[0]


if __name__ == "__main__":
    test_cases = [
        ("leetcode", ["leet", "code"], True),           # Simple split
        ("applepenapple", ["apple", "pen"], True),      # Word reuse
        ("catsandog", ["cats", "dog", "sand", "and", "cat"], False),  # No valid split
        ("cars", ["car", "ca", "rs"], True),            # Overlapping: ca + rs
        ("aaaaaaa", ["aaaa", "aaa"], True),             # 4 + 3 = 7
        ("a", ["a"], True),                             # Single char
        ("", ["a"], True),                              # Empty string
        ("abcd", ["a", "b", "c"], False),               # Missing 'd'
    ]
    
    print("="*55)
    print("WORD BREAK (Dynamic Programming)")
    print("="*55)
    
    for s, word_dict, expected in test_cases:
        result = word_break(s, word_dict)
        status = "✓" if result == expected else "✗"
        display_s = f'"{s}"' if s else '""'
        print(f"{status} s={display_s:20} words={word_dict}")
        print(f"  Expected: {expected}, Got: {result}")
