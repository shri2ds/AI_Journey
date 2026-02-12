def rectangular_matrix(matrix):
     """ Rotate the M×N matrix """
    if not matrix:
        return []

    rows, columns = len(matrix), len(matrix[0])
    result_matrix = [[0] * rows for _ in range(columns)]
    for r in range(rows):
        for c in range(columns):
            result_matrix[c][rows-1-r] = matrix[r][c]

    return result_matrix

  
if __name__ == "__main__":
    # Simple example
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ]
    print(f"Original matrix is {matrix}")
    rectangular_matrix(matrix)
    print(f"Rotated Rectangular matrix is {matrix}")          
