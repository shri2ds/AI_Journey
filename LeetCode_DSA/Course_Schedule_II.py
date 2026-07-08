from typing import List


class CourseSchedule:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        HashMap = { i:[] for i in range(numCourses) }
        for cor, pre in prerequisites:
            HashMap[cor].append(pre)

        visited = set()   # Fully processed courses
        cycle = set()     # Currently in DFS path (for cycle detection)
        result = []

        def dfs(cor):
            if cor in cycle:      # Cycle detected!
                return False
            if cor in visited:    # Already processed, skip
                return True

            cycle.add(cor)        

            for pre in HashMap[cor]:  # Process all prerequisites first
                if not dfs(pre):
                    return False      # Cycle found in prerequisite

            cycle.remove(cor)     # Done with this path
            visited.add(cor)      # Mark as fully processed
            result.append(cor)    # Add to result AFTER prerequisites
            return True

        for cor in range(numCourses):
            if not dfs(cor):
                return []         # Cycle detected, impossible

        return result

if __name__ == "__main__":
    sol = CourseSchedule()
    
    test_cases = [
        # (numCourses, prerequisites, expected_options, description)
        (4, [[1, 0], [2, 0], [3, 1], [3, 2]], [[0, 1, 2, 3], [0, 2, 1, 3]], "DAG: Valid dependency graph"),
        (3, [[0, 1], [1, 2], [2, 0]], [[]], "Loop: Cycle 0 → 1 → 2 → 0"),
        (2, [[1, 0]], [[0, 1]], "Simple: Course 1 requires Course 0"),
    ]
    
    print("Course Schedule II - Test Results")
    print("=" * 60)
    
    all_passed = True
    for i, (num, prereqs, expected_options, desc) in enumerate(test_cases, 1):
        result = sol.findOrder(num, prereqs)
        passed = result in expected_options
        status = "✓" if passed else "✗"
        if not passed:
            all_passed = False
        print(f"{status} Test {i}: {desc}")
        print(f"   Courses: {num}, Prerequisites: {prereqs}")
        print(f"   Expected: {expected_options[0] if len(expected_options) == 1 else 'one of ' + str(expected_options)}")
        print(f"   Got: {result}")
        print()
    
    print("=" * 60)
    print(f"Result: {'All tests passed!' if all_passed else 'Some tests failed.'}")
