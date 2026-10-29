from collections import deque
from typing import List

class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList: return 0
        wordSet = set(wordList)
        if beginWord in wordSet: wordSet.remove(beginWord)
        m = len(beginWord)
        alphabets = 'abcdefghijklmnopqrstuvwxyz'

        queue = deque()
        queue.append((beginWord, 1))
        while queue:
            word, number = queue.popleft()
            if word == endWord: return number
            for i in range(m):
                for letter in alphabets:
                    if word[i] == letter: continue
                    new_word = word[:i] + letter + word[i+1:]
                    if new_word in wordSet:
                        wordSet.remove(new_word)
                        queue.append((new_word, number+1))

        return 0


if __name__ == "__main__":
    sol = Solution()
    assert sol.ladderLength("hit", "cog", ["hot","dot","dog","lot","log","cog"]) == 5
    assert sol.ladderLength("hit", "cog", ["hot","dot","dog","lot","log"]) == 0
    assert sol.ladderLength("a", "c", ["a", "b", "c"]) == 2
    assert sol.ladderLength("a", "c", ["b"]) == 0
    print("All test cases passed!")
