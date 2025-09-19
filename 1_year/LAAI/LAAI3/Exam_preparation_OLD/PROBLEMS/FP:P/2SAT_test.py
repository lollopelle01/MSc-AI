def simple_2sat_solver(L):
    n = len(L)
    for i in range(n):
        a_i = L[i][0]
        b_i = L[i][1]
        valid = True
        for j in range(n):
            a_j = L[j][0]
            b_j = L[j][1]
            if (a_i != a_j) and (b_i != b_j):
                valid = False
                break
        if valid:
            return (a_i, b_i)
    return (-1, -1)

if __name__ == "__main__":
    L = [(1, 0), (0, 1), (0, 0)]
    print(simple_2sat_solver(L))  # Output: (0, 0)
