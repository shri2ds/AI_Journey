from typing import List

class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        rows, columns = len(matrix), len(matrix[0])
        rowZero = False

        for r in range(rows):
            for c in range(columns):
                if matrix[r][c] == 0:
                    matrix[0][c] = 0
                    if r > 0:
                        matrix[r][0] = 0
                    else:
                        rowZero = True

        for r in range(1, rows):
            for c in range(1, columns):
                if matrix[r][0] == 0 or matrix[0][c] == 0:
                    matrix[r][c] = 0

        if rowZero:
            for c in range(columns):
                matrix[0][c] = 0

        if matrix[0][0] == 0 :
            for r in range(rows):
                matrix[r][0] = 0

if __name__ == "__main__":
    #Example usage
    sol = Solution()
    matrix = [[1],[0],[3]]
    print("Original:", matrix)
    sol.setZeroes(matrix)
    print("Modified:", matrix)
