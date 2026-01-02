def isPalindrome(x):
    """
    :type x: int
    :rtype: bool
    """
    if x < 0:
        return False
    if x % 10 == 0 and x != 0:
        return False

    original_num = x
    iteration = 0
    reverse_number = 0

    while original_num > 0:
        number = original_num % 10
        reverse_number = reverse_number * 10 + number
        original_num //= 10
        iteration += 1
    print(f"Reverse number is {reverse_number}")

    if x == reverse_number:
        return True
    else:
        return False

if __name__ == "__main__":
    # Example usage
    x = 123
    result = isPalindrome(x)
    print(result)
