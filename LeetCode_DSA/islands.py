from collections import deque

def numIslands(grid):
    """Count the number of islands (connected groups of '1's) in a binary grid."""
    if not grid:
      return 0

    rows, cols = len(grid), len(grid[0])
    visited = set()  # track cells already explored
    islands = 0

    def dfs(r,c):
        """Mark all cells in the current island starting from (r, c)."""
        q = deque()
        visited.add((r,c))
        q.append((r,c))

        while q:
            row, col = q.popleft()
            directions = [ [-1, 0], [1, 0], [0, 1], [0, -1] ]   # four-directional neighbors
            for dr, dc in directions:
                r, c = dr + row, dc + col
                if r in range(rows) and c in range(cols) and grid[r][c] == "1" and (r,c) not in visited:
                    q.append((r,c))
                    visited.add((r,c))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and (r,c) not in visited:
              dfs(r,c)
              islands += 1

    return islands

if __name__ == "__main__":
    #Example usage
    grid = [
        ["1","1","1","1","0"],
        ["1","1","0","1","0"],
        ["1","1","0","0","0"],
        ["0","0","0","0","0"]
        ]
    result = numIslands(grid)
    print(result)
  
