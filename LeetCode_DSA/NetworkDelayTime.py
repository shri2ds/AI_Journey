from typing import List
import heapq

class NetworkDelayTime:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        edges = {i:[] for i in range(1, n+1)}
        for s, d, w in times:
            edges[s].append((d, w))

        visited = set()
        t = 0
        minHeap = [(0, k)]    # Min-Heap stores elements as: (cumulative_path_time, target_node)

        while minHeap:
            w1, n1 = heapq.heappop(minHeap)
            if n1 in visited:
                continue

            visited.add(n1)
            t = max(t, w1)

            for n2, w2 in edges[n1]:
                if n2 not in visited:
                    heapq.heappush(minHeap, (w1 + w2, n2))

        return t if len(visited) == n else -1 

if __name__ == "__main__":
    sol = NetworkDelayTime()

    # Test Case 1: Standard routing delay paths
    times1 = [[2, 1, 1], [2, 3, 1], [3, 4, 1]]
    assert sol.networkDelayTime(times1, 4, 2) == 2

    # Test Case 2: Direct path validation
    times2 = [[1, 2, 1]]
    assert sol.networkDelayTime(times2, 2, 1) == 1

    # Test Case 3: Disconnected graph partitions
    times3 = [[1, 2, 1]]
    assert sol.networkDelayTime(times3, 2, 2) == -1

    print(" All Network Delay Time Dijkstra test assertions passed cleanly!")
