from typing import List

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        PreMap = { i:[] for i in range(numCourses)}
        for cor, pre in prerequisites:
            PreMap[cor].append(pre)

        visited = set()
        def dfs(cor):
            if cor in visited:
                return False
            if PreMap[cor] == []:
                return True

            visited.add(cor)
            for pre in PreMap[cor]:
                if not dfs(pre): return False

            visited.remove(cor)
            PreMap[cor] = []
            return True

        for cor in range(numCourses):
            if not dfs(cor): return False

        return True

if __name__ == "__main__":
    sol = Solution()
    
    test_cases = [
        # (numCourses, prerequisites, expected, description)
        (4, [[1, 0], [2, 0], [3, 1], [3, 2]], True, "DAG: Valid dependency graph"),
        (3, [[0, 1], [1, 2], [2, 0]], False, "Loop: Cycle 0 → 1 → 2 → 0"),
        (4, [[1, 0], [3, 2]], True, "Disconnected: Two separate chains"),
    ]
    
    print("Course Schedule - Test Results")
    print("=" * 60)
    
    all_passed = True
    for i, (num, prereqs, expected, desc) in enumerate(test_cases, 1):
        result = sol.canFinish(num, prereqs)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"{status} Test {i}: {desc}")
        print(f"   Courses: {num}, Prerequisites: {prereqs}")
        print(f"   Expected: {expected}, Got: {result}")
        print()
    
    print("=" * 60)
    print(f"Result: {'All tests passed!' if all_passed else 'Some tests failed.'}")
