from typing import List

class SafeNodes:
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        hashmap = {}
        result = []
        def dfs(i):
            if i in hashmap:
                return hashmap[i]
            hashmap[i] = False
            for neigh in graph[i]:
                if not dfs(neigh):
                    return hashmap[i]
            hashmap[i] = True
            return True

        for i in range(len(graph)):
            if dfs(i):
                result.append(i)
                
        return result

if __name__ == "__main__":
    sol = SafeNodes()
    
    test_cases = [
        # (graph, expected, description)
        ([[1,2],[2,3],[5],[0],[5],[],[]], [2,4,5,6], "Example 1: Graph with cycles and safe nodes"),
        ([[1,2,3,4],[1,2],[3,4],[0,4],[]], [4], "Example 2: Linear graph with one sink"),
        ([[1],[],[1],[]], [1,2,3], "Example 3: Disconnected components"),
    ]
    
    print("Safe Nodes - Test Results")
    print("=" * 60)
    
    all_passed = True
    for i, (graph, expected, desc) in enumerate(test_cases, 1):
        result = sol.eventualSafeNodes(graph)
        passed = result == expected
        status = "✓" if passed else "✗"
        if not passed:
            all_passed = False
        print(f"{status} Test {i}: {desc}")
        print(f"   Graph: {graph}")
        print(f"   Expected: {expected}")
        print(f"   Got: {result}")
        print()
    
    print("=" * 60)
    if all_passed:
        print("Result: All tests passed!")
    else:
        print("Result: Some tests failed!")

