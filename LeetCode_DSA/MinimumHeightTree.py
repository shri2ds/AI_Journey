from typing import List

class MHT:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n < 2:
            return [x for x in range(n)]

        neighbors = [set() for x in range(n)]
        for start, end in edges:
            neighbors[start].add(end)
            neighbors[end].add(start)

        leaves = []
        for i in range(n):
            if len(neighbors[i]) == 1:
                leaves.append(i)

        remaining_nodes = n
        while remaining_nodes > 2:
            remaining_nodes -= len(leaves)
            temp = []

            for leaf in leaves:
                for neighbor in neighbors[leaf]:
                    neighbors[neighbor].remove(leaf)
                    if len(neighbors[neighbor]) == 1:
                        temp.append(neighbor)
            leaves = temp

        return leaves

if __name__ == "__main__":
    sol = MHT()

    # Test Case 1: Standard Tree (Single Centroid)
    n1 = 4
    edges1 = [[1, 0], [1, 2], [1, 3]]
    # Expected Output: [1]
    print("Test 1:", sol.findMinHeightTrees(n1, edges1))

    # Test Case 2: Tree with Two Centroids
    n2 = 6
    edges2 = [[3, 0], [3, 1], [3, 2], [3, 4], [5, 4]]
    # Expected Output: [3, 4]
    print("Test 2:", sol.findMinHeightTrees(n2, edges2))

    # Test Case 3: Edge Case - Single Node (No Edges)
    n3 = 1
    edges3 = []
    # Expected Output: [0]
    print("Test 3:", sol.findMinHeightTrees(n3, edges3))

    # Test Case 4: Straight Line Tree (Path Graph)
    n4 = 2
    edges4 = [[0, 1]]
    # Expected Output: [0, 1]
    print("Test 4:", sol.findMinHeightTrees(n4, edges4))
