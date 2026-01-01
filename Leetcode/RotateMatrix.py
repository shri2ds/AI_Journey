def rotate(matrix):
  """ Rotate the N×N matrix 90° clockwise in-place """
  n = len(matrix)
  for i in range(n):
      for j in range(i+1, n):
        matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

  for row in matrix:
      row.reverse()

if __name__ == "__main__":
    # Simple example
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    print(f"Original matrix is {matrix}")
    rotate(matrix)
    print(f"Rotated matrix is {matrix}")
