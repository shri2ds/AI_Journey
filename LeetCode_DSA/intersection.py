def intersection(nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        # result = []
        # for i in nums1:
        #     for j in nums2:
        #         if i == j:
        #             result.append(i)
        #
        # result_set = set(result)
        # return list(result_set)
        ''' The Optimised code '''
        set1 = set(nums1)
        set2 = set(nums2)
        return list(set1 & set2)

if __name__ == "__main__":
    # Example usage
    x = [1,2,2,1]
    y = [2,2]
    result = intersection(x, y)
    print(result)
