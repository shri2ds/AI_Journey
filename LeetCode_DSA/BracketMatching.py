def bracket_matching(s):
    """
    Check if brackets in string are properly balanced and nested.
    
    Valid brackets: (), {}, []
    
    Time: O(n) - single pass through string
    Space: O(n) - worst case all opening brackets
    
    Args:
        s: String containing brackets
    
    Returns:
        True if balanced, False otherwise
    """
    stack = []
    bracket_map = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in '({[':
            stack.append(char)
        elif char in ')}]':
            if not stack or stack[-1] != bracket_map[char]:
                return False
            stack.pop()
    
    return len(stack) == 0


if __name__ == "__main__":
    test_cases = [
        ("{[()()]}", True),    # Properly nested
        ("([)()]", False),     # Crossed brackets
        ("()", True),          # Simple pair
        ("(", False),          # Unclosed
        (")", False),          # Closing with nothing open
        ("", True),            # Empty string
        ("((()))", True),      # Deep nesting
        ("({[]})", True),      # Mixed types nested
        ("[(])", False),       # Wrong order closing
        ("{[]}", True),        # Sequential inside
        ("}{", False),         # Starts with closing
    ]
    
    print("="*50)
    print("BRACKET MATCHING TEST CASES")
    print("="*50)
    
    for s, expected in test_cases:
        result = bracket_matching(s)
        status = "✓" if result == expected else "✗"
        display = f'"{s}"' if s else '""'
        print(f"{status} Input: {display:15} Expected: {expected}, Got: {result}")
