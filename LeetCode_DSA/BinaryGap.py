def BinaryGap(n):
    binary = bin(n)[2:]  # Convert to binary and remove '0b' prefix
    current_gap = 0
    max_gap = 0
  
    for bit in binary:  
        if bit == "1":
            max_gap = max(max_gap, current_gap)
            current_gap = 0
        else:
            current_gap += 1

    return max_gap

if __name__ == "__main__":
    print(BinaryGap(9))   # Expected: 2

