def twoSum(nums, target):
    num_to_index = {}
    for i, num in enumerate(nums):
        complement = target - num
        print(f"Complement - {complement}, num - {num}")
        if complement in num_to_index:
            return [num_to_index[complement], i]
        num_to_index[num] = i
        print(f"num_to_index - {num_to_index}")

if __name__ == "__main__":
    # Example usage
    nums = [2, 7, 11, 15]
    target = 9
    result = twoSum(nums, target)
    print(result)
