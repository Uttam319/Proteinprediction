def linear_partition(sequence):
    n = len(sequence)
    INF = float('inf')

    # Initialize matrices
    q = [0] * (n + 1)
    q[0] = 1
    q[n] = n
    p = [INF] * (n + 1)
    p[0] = 0
    b = [0] * (n + 1)

    # Calculate q values
    for j in range(1, n + 1):
        for i in range(j, 0, -1):
            min_val = INF
            min_k = 0
            for k in range(j, i - 1, -1):
                if q[k - 1] + 1 < min_val:
                    min_val = q[k - 1] + 1
                    min_k = k
            q[j] = min_val
            b[j] = min_k

    # Calculate p values
    for j in range(1, n + 1):
        for k in range(b[j], j + 1):
            p[j] = min(p[j], p[k - 1] + (j - k + 1) ** 2)

    # Traceback
    breakpoints = []
    j = n
    while j > 0:
        k = b[j]
        breakpoints.append(k - 1)
        j = k - 1

    return breakpoints


def dot_bracket_from_partition(sequence, breakpoints):
    structure = ['.'] * len(sequence)
    for i in range(len(breakpoints) - 1):
        left = breakpoints[i]
        right = breakpoints[i + 1]
        structure[left] = '('
        structure[right] = ')'
    return ''.join(structure)


def name_rna_structure(structure):
    if "()" in structure:
        return "Hairpin Loop"
    elif "." in structure:
        return "Single-stranded"
    elif "(" in structure and ")" in structure:
        return "Stem Loop"
    else:
        return "Unknown"



# Example usage
sequence = input("Enter an RNA sequence: ")
breakpoints = linear_partition(sequence)
predicted_structure = dot_bracket_from_partition(sequence, breakpoints)
print("Predicted RNA structure:", predicted_structure)
print("RNA Structure Name:", name_rna_structure(predicted_structure))
