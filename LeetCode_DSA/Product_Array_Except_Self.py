def product_array_except_self(nums):
    """
    :type nums: List[int]
    """
    output = [1] * len(nums)
    # PASS 1: Build left products
    prefix = 1
    for i in range(len(nums)):
        output[i] = prefix
        prefix *= nums[i]
    # PASS 2: Multiply by right products
    suffix = 1
    for i in range(len(nums)-1, -1, -1):
        output[i] *= suffix
        suffix *= nums[i]
    return output

if __name__ == "__main__":
    #Example usage
    nums = [-2, -1, 0, 1, 2]
    result = product_array_except_self(nums)
    print(result)
