from typing import List


class Solution:
    """Provides an interval-merging helper compatible with LeetCode's API."""

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        Args:
            intervals : List[List[int]]
                List of half-open intervals ``[start, end]`` with ``start <= end``.
            List[List[int]]
                Intervals with all overlaps collapsed into single segments.
        """

        # 1. Sort by left element of the list within intervals
        intervals.sort(key=lambda x: x[0])

        merged = []
        left, right = intervals[0][0], intervals[0][1]
        merged.append(intervals[0])
        intervals.pop(0)

        # Walk remaining intervals and merge whenever there is an overlap.
        for interval in intervals:
            if interval[0] <= right:
                # Overlap: extend the current window to cover the furthest end.
                merged.pop(-1)
                right = max(right, interval[1])
                merged.append([left, right])
            else:
                # No overlap: start tracking a brand-new window.
                left = interval[0]
                right = interval[1]
                merged.append(interval)

        return merged


if __name__ == "__main__":
    # Example: intervals [1,4] and [2,3] collapse into [1,4].
    input_intervals = [[1, 4], [2, 3]]
    sol = Solution()
    merged_result = sol.merge(input_intervals)
    print(merged_result)
