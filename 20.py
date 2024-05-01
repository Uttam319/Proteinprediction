
def detect_structure(structure):
    if '()' in structure:
        if '().()' in structure:
            return "Stem-loop"
        else:
            return "Hairpin-loop"
    elif '(())' in structure:
        return "Stem-loop"
    elif '.()' in structure or '().' in structure:
        return "Internal loop"
    elif '.' in structure:
        return "Single-stranded"
    else:
        return "Unknown"

def linear_fold(seq):
    n = len(seq)
    
    dp = [[0] * n for _ in range(n)]
    traceback = [[0] * n for _ in range(n)]

    for length in range(1, n):
        for i in range(n - length):
            j = i + length
            candidates = [(dp[i][k] + dp[k+1][j], k) for k in range(i, j)]
            best_score, traceback_point = max(candidates)
            dp[i][j] = best_score
            traceback[i][j] = traceback_point
           
            if seq[i] == 'A' and seq[j] == 'U' or seq[i] == 'U' and seq[j] == 'A' or seq[i] == 'C' and seq[j] == 'G' or seq[i] == 'G' and seq[j] == 'C':
                if dp[i][j] < dp[i+1][j-1] + 1:
                    dp[i][j] = dp[i+1][j-1] + 1
                    traceback[i][j] = -1
    
    def traceback_structure(i, j):
        if i >= j:
            return ""
        if traceback[i][j] == -1:
            return "(" + traceback_structure(i+1, j-1) + ")"
        else:
            k = traceback[i][j]
            return traceback_structure(i, k) + "." + traceback_structure(k+1, j)
    return traceback_structure(0, n-1)

if __name__ == "__main__":
    seq = input("Enter RNA sequence: ").upper()
    predicted_structure = linear_fold(seq)
    print("Predicted secondary structure:", predicted_structure)
    
    structure_type = detect_structure(predicted_structure)
    print("Predicted structure type:", structure_type)
