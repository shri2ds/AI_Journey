from collections import defaultdict

def groupAnagrams(strs: List[str]) -> List[List[str]]:
    output = defaultdict(list)

    for word in strs:
        alp_count = [0] * 26
        for letter in word:
            alp_count[ord(letter) - ord("a")] += 1
        output[tuple(alp_count)].append(word)
    
    return list(output.values())

if __name__ == "__main__":
    # Example usage
    strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    result = groupAnagrams(strs)
    print(result)
