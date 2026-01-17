from typing import List


class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix:
            return []

        result = []
        left, right = 0, len(matrix[0])
        top, bottom = 0, len(matrix)

        while left < right and top < bottom:
            # Moving across the top row
            for i in range(left, right):
                result.append(matrix[top][i])
            top += 1

            # Moving across the right column
            for i in range(top, bottom):
                result.append(matrix[i][right-1])
            right -= 1

            if not (left < right and top < bottom):
                break

            # Moving across the bottom row
            for i in range(right-1, left-1, -1):
                result.append(matrix[bottom-1][i])
            bottom -= 1

            # Moving across the left column
            for i in range(bottom-1, top-1, -1): #(1,0) (2,0)
                result.append(matrix[i][left])
            left += 1

        return result

  if __name__ == "__main__":
    s = Solution()
    print(s.spiralOrder( [[1,2,3,4],
                          [5,6,7,8],
                          [9,10,11,12]] ))
    
