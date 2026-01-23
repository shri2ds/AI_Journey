from typing import List

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        """ Return index of target within rotated sorted nums or -1 if missing. """

        left, right = 0, len(nums) - 1

        while left <= right:
            middle = (left + right) // 2

            if nums[middle] == target:
                return middle

            # Determine which half is sorted
            if nums[left] <= nums[middle]:
                # Left half is sorted
                if nums[left] <= target < nums[middle]:
                    right = middle - 1
                else:
                    left = middle + 1
            else:
                # Right half is sorted
                if nums[middle] < target <= nums[right]:
                    left = middle + 1
                else:
                    right = middle - 1

        return -1


if __name__ == "__main__":
    nums = [6, 7, 8, 1, 2, 3, 4, 5]
    target = 8
    result = Solution().search(nums, target)
    print(result)
