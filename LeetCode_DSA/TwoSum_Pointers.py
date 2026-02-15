def TwoSum_Pointers(nums, target):
    """ 
        Two Pointer technique to solve Two Sum Problem
        Args:
            numbers (List[int]): A 1-indexed array of integers sorted in non-decreasing order.
            target (int): The target sum.
        Returns:
            List[int]: A list containing the indices of the two numbers (1-based).
    """  
    left, right = 0, len(nums)-1
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum < target:
            left += 1
        elif current_sum > target:
            right -= 1
        else:
            return [left+1, right+1]    


if __name__ == "__main__":
    # Example usage
    nums = [2, 7, 11, 15]
    target = 9
    result = twoSum(nums, target)
    print(result)
