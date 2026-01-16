from typing import List

class Solution:
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        """
        :param mat: Input 2D matrix (List of Lists)
        :param r: Target number of rows
        :param c: Target number of columns
        :return: Reshaped matrix OR original matrix if reshape is not possible
        """

        rows, columns = len(mat), len(mat[0])
        if (r * c) != (rows * columns):
          return mat

        result = [[0 for _ in range(c)] for _ in range(r)]
        val = 0
        for i in range(rows):
            for j in range(columns):
                result[val // c][val % c] = mat[i][j]
                val += 1
        return result


if __name__ == "__main__":
    solver = Solution()

    # Test 1: Standard Reshape
    # Input: [[1,2], [3,4]], Target: (1, 4)
    # Expected: [[1, 2, 3, 4]]
    res1 = solver.matrixReshape([[1, 2], [3, 4]], 1, 4)
    print(res1)

    # Test 2: Impossible Reshape
    # Input: [[1,2], [3,4]], Target: (2, 4) (Needs 8 elements, have 4)
    # Expected: [[1, 2], [3, 4]] (Original)
    res2 = solver.matrixReshape([[1, 2], [3, 4]], 2, 4)
    print(res2)
