from typing import List
from collections import deque

class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        fresh, time = 0, 0
        q = deque()

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    fresh += 1
                if grid[i][j] == 2:
                    q.append([i, j])

        directions = [[-1,0], [1,0], [0,1], [0,-1]]
        while q and fresh > 0:
            for i in range(len(q)):
                r, c = q.popleft()
                for dr, dc in directions:
                    row, col = r + dr, c + dc
                    if (row == len(grid) or row < 0 or
                        col == len(grid[0]) or col < 0 or
                        grid[row][col] != 1 ):
                        continue

                    q.append([row, col])
                    grid[row][col] = 2
                    fresh -= 1

            time += 1

        return time if fresh == 0 else -1

if __name__ == "__main__":
    sol = Solution()

    # --- Test Case 1: Standard decay trajectory ---
    # 2 = Rotten, 1 = Fresh, 0 = Empty cell
    grid1 = [
        [2, 1, 1],
        [1, 1, 0],
        [0, 1, 1]
    ]

    print("Running Test Case 1...")
    assert sol.orangesRotting(grid1) == 4
    print("Test Case 1 Passed!")

    # --- Test Case 2: Blocked/Isolated fresh orange ---
    grid2 = [
        [2, 1, 1],
        [0, 1, 1],
        [1, 0, 1]
    ]
  
    print("\nRunning Test Case 2...")
    assert sol.orangesRotting(grid2) == -1
    print("Test Case 2 Passed!")

    # --- Test Case 3: Zero fresh oranges ---
    grid3 = [
        [0, 2]
    ]
    print("\nRunning Test Case 3...")
    assert sol.orangesRotting(grid3) == 0
    print("Test Case 3 Passed!")
