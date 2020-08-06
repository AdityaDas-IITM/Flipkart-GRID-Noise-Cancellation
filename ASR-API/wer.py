def wer(r, h):
    """
    Calculation of WER with Levenshtein distance.
    O(nm) time and space complexity.

    Parameters
    ----------
    r - string - reference string
    h - string - test string

    Returns
    -------
    float - Word Error Rate as a percentage

    Examples
    --------
    wer("who is there", "is there") returns 1
    wer("who is there", "") returns 3
    wer("", "who is there") returns 3
    """

    # Split string into list of words
    r = r.split()
    h = h.split()

    # Dynamic programming matrix
    d = [[0 for j in range(len(h)+1)] for i in range(len(r)+1)]
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation of values in matrix
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    # returning as percentage
    return round(d[len(r)][len(h)] * 100/float(len(r)), 2)

#print(wer("बेकार में बड़ा टी शर्ट डाल दो कार्ट में", "बेकार में बड़ा टी शर्ट डाल दो फिटनेस में"))

if __name__ == "__main__":
    a = input("Enter reference string: ")
    b = input("Enter test string: ")
    print(wer(a,b))
