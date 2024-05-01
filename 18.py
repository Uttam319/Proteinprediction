import numpy as np

def linear_fold(sequence):
    n = len(sequence)
    INF = float('inf')

    # Initialize matrices
    q = np.zeros((n+1, n+1))
    p = np.zeros((n+1, n+1))
    b = np.zeros((n+1, n+1))

    # Initialize parameters
    for i in range(n):
        q[i, i+1] = 0
        p[i, i+1] = INF

    # Dynamic programming
    for d in range(1, n):
        for i in range(n - d):
            j = i + d + 1
            min_val = INF
            min_k = -1
            for k in range(i + 1, j):
                q_val = q[i, k] + q[k, j]
                if q_val < min_val:
                    min_val = q_val
                    min_k = k
            q[i, j] = min_val
            p[i, j] = min([p[i, k] + p[k, j] for k in range(i + 1, j)], default=INF)
            b[i, j] = min_k

    # Traceback to get the secondary structure
    def traceback(i, j):
        if j <= i + 1:
            return
        k = int(b[i, j])
        if k == -1:
            # Unpaired base
            structure.append('.')
            traceback(i + 1, j)
        else:
            # Paired bases
            structure.append('(')
            structure.append(')')
            traceback(i, k)
            traceback(k, j)

    structure = []
    traceback(0, n)

    return ''.join(structure)

def name_rna_structure(structure):
    if "()" in structure:
        return "Hairpin Loop"
    elif "." in structure:
        return "Single-stranded"
    elif "(" in structure and ")" in structure:
        if structure.startswith("(") and structure.endswith(")"):
            return "Stem Loop"
        else:
            return "Internal Loop"
    else:
        return "Unknown"


# User input
sequence = input("Enter an RNA sequence: ")

# Predict secondary structure
predicted_structure = linear_fold(sequence)
print("Predicted RNA structure:", predicted_structure)
print("RNA Structure Name:", name_rna_structure(predicted_structure))

