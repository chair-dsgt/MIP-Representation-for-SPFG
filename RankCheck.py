import gurobipy as gp
import math
import numpy as np

# %%
def getVerdict(mat, matAug, col):
    """
    Function that compares the rank of the matrix Q with the rank of the augmented matrix
    Q|b. Returns the rank when there is at least one solution, or stop the code otherwise.
    """
    a = np.array(mat)
    b = np.array(matAug)
    rank_a = np.linalg.matrix_rank(a)
    rank_b = np.linalg.matrix_rank(b)
    n = len(mat[0])
    verdict = ""
    if rank_b > rank_a:
      verdict = "No solution."
    elif rank_a == rank_b:
      if rank_a == n:
        verdict = "Unique solution."
      elif rank_a < n:
        verdict = "Infinity of solutions."

    print(rank_a, rank_b, verdict)
    assert rank_a == rank_b, "Trying to construct A, but no solution for column " + str(col)
    return rank_b.item()

def performRankTest(dico: dict, N: int):
    """
    Function that constructs Q, b and Q|b for the Qx = b linear equation systems for each column of A.
    Performs rank test, that can stop the code if the sym PFG is not writable in matrix A form.
    Return a list of [n, Q, b, rank] that can then be used to construct A.
    """
    constructA_material = []
    for n in range(1, N+1):
        Q_lst = []
        b_lst = []
        Qb_lst = []
        for (x, v) in dico.items():
            if x[0] == n:
                # This dico entry concerns current coal size n
                q = [x[1][j] if j < len(x[1]) else 0 for j in range(N-n+1)]
                Q_lst.append(q)
                b_lst.append(v)
                Qb_lst.append(q + [v])
        rank = getVerdict(Q_lst, Qb_lst, n)
        if len(Q_lst[0]) < rank:
            padding = [0 for i in range(rank - len(Q_lst[0]))]
            for e in Q_lst : e += padding
        constructA_material.append([n, Q_lst, b_lst, rank])
    return constructA_material

def createColumnModel(Q, b, rank):
    """
    Function that creates and solve the Qv = b model for a column of A. Returns an array containing values for
    this column.
    """
    M = gp.Model("GenColA")
    X = M.addVars(rank,vtype=gp.GRB.CONTINUOUS,name="X",lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
    M.setObjective(X.sum('*'),gp.GRB.MAXIMIZE)
    print(Q, b)
    M.addConstrs((b[i] == gp.quicksum(Q[i][j]*X[j] for j in range(rank)) for i in range(len(b))),name="Qvb")
    M.Params.OutputFlag = 0
    M.optimize()
    assert M.SolCount > 0, "No more solutions for matrix A in createColumnModel, means there is an error."
    return [0 if math.isclose(X[i].X, 0, rel_tol=1e-5) else X[i].X for i in range(rank)]

def dictToMat(dico: dict, N: int):
    """
    This function creates models for every column of A and optimize them in order to retrieve a square matrix A
    given a dictionary of a symmetric PFG.
    """
    A = np.zeros((N,N))
    column_material = performRankTest(dico, N)
    assert len(column_material) == N, "Missing material to constitute matrix A."
    for material in column_material:
        # Create and optimize model.
        col = createColumnModel(material[1], material[2], material[3])
        # Copy info in A.
        for i in range(material[3]):
            A[i,material[0]-1] = col[i]
    return A

def FindQAndb(dico: dict, N: int):
    """
    Function that constructs Q and b, that can then be used to construct A using the least squares method.
    """
    constructA_material = []
    for n in range(1, N+1):
        Q_lst = []
        b_lst = []
        for (x, v) in dico.items():
            if x[0] == n:
                # This dico entry concerns current coal size n
                q = [x[1][j] if j < len(x[1]) else 0 for j in range(N-n+1)]
                Q_lst.append(q)
                b_lst.append(v)
        constructA_material.append([n, Q_lst, b_lst])
    return constructA_material

def dictToMatUsingLeastSquares(dico: dict, N: int):
    """
    This function finds the appropriate values for every column of A, which are the solutions to small linear
    systems Qx = b. The matrix A might not be exact in places where the column does not have a solution, but we
    hope it might constitute an approximation close enough to drag interesting conclusions about the symm. PFG.
    """
    A = np.zeros((N,N))
    R = []
    column_material = FindQAndb(dico, N)
    assert len(column_material) == N, "Missing material to constitute matrix A with least squares method."
    for material in column_material:
        # Generate the least squares solution.
        Q = np.array(material[1])
        b = np.array(material[2])
        col, sum_of_sq_residuals, lencol, arr = np.linalg.lstsq(Q, b, rcond=None)
        # Little if to print coefficient of determination, helping us to see what approximations are good.
        if len(sum_of_sq_residuals) == 0:
            if len(Q[0]) > 1:
                print("For column studying the coalition of", Q[0][1], "players, R^2 is 1")
            elif len(Q[0]) == 1:
                print("For column studying the coalition of", Q[0][0], "players, R^2 is 1")
            R.append(1)
        elif len(sum_of_sq_residuals) == 1:
            if sum_of_sq_residuals < 0.0000001:
                print("For column studying the coalition of", Q[0][1], "players, R^2 is 1")
                R.append(1)
            else:
                b_bar = np.mean(b)
                tss = np.sum((b - b_bar)**2)
                R_squared = 1 - (sum_of_sq_residuals / tss)
                print("For column studying the coalition of", Q[0][1], "players, R^2 is", round(R_squared[0], 3))
                R.append(round(R_squared[0], 3))
        else:
            print("Weird sum of sq residuals, is there an error somewhere?")
        # Copy info in A.
        for i in range(col.shape[0]):
            A[i,material[0]-1] = col[i]
    return A, R

# %%
if __name__ == "__main__":
    """ mat = [
      [5,2,0,0,0,0],
      [4,2,1,0,0,0],
      [3,2,2,0,0,0],
      [3,2,1,1,0,0],
      [2,2,2,1,0,0],
      [2,2,1,1,1,0],
      [2,1,1,1,1,1]
    ]

    matAug = [
      [5,2,0,0,0,0,19],
      [4,2,1,0,0,0,27],
      [3,2,2,0,0,0,39],
      [3,2,1,1,0,0,47],
      [2,2,2,1,0,0,59],
      [2,2,1,1,1,0,7],
      [2,1,1,1,1,1,7]
    ]

    getVerdict(mat, matAug, 2)

    mat = [
        [7, 2, 0, 0, 0, 0, 0, 0],
        [6, 2, 1, 0, 0, 0, 0, 0],
        [5, 2, 2, 0, 0, 0, 0, 0],
        [4, 3, 2, 0, 0, 0, 0, 0],
        [5, 2, 1, 1, 0, 0, 0, 0],
        [4, 2, 2, 1, 0, 0, 0, 0],
        [3, 3, 2, 1, 0, 0, 0, 0],
        [3, 2, 2, 2, 0, 0, 0, 0],
        [4, 2, 1, 1, 1, 0, 0, 0],
        [3, 2, 2, 1, 1, 0, 0, 0],
        [2, 2, 2, 2, 1, 0, 0, 0],
        [3, 2, 1, 1, 1, 1, 0, 0],
        [2, 2, 2, 1, 1, 1, 0, 0],
        [2, 2, 1, 1, 1, 1, 1, 0],
        [2, 1, 1, 1, 1, 1, 1, 1]
    ]

    mat = [
        [2,1,1,1,1,1],
        [2,2,1,1,1,0],
        [2,2,2,1,0,0],
        [3,2,1,1,0,0],
        [3,2,2,0,0,0],
        [4,2,1,0,0,0],
        [5,2,0,0,0,0],
    ]

    mat = [
        [2,1,1,1,1,1,1],
        [2,2,1,1,1,1,0],
        [3,2,1,1,1,0,0],
        [2,2,2,1,1,0,0],
        [3,2,2,1,0,0,0],
        [2,2,2,2,0,0,0],
        [4,2,1,1,0,0,0],
        [3,3,2,0,0,0,0],
        [4,2,2,0,0,0,0],
        [5,2,1,0,0,0,0],
        [6,2,0,0,0,0,0],
    ]

    mat = [
        [1,1,1,1,1,1,1],
        [2,1,1,1,1,1,0],
        [2,2,1,1,1,0,0],
        [3,1,1,1,1,0,0],
        [2,2,2,1,0,0,0],
        [3,2,1,1,0,0,0],
        [4,1,1,1,0,0,0],
        [3,3,1,0,0,0,0],
        [4,2,1,0,0,0,0],
        [5,1,1,0,0,0,0],
        [6,1,0,0,0,0,0],
    ]

    a = np.array(mat)
    rank_a = np.linalg.matrix_rank(a)
    print(rank_a) """

    N = 8
    dictionary = dict()

    # One and two
    dictionary[tuple([1, tuple([1, 1, 1, 1, 1, 1, 1, 1])])] = 5
    dictionary[tuple([1, tuple([2, 1, 1, 1, 1, 1, 1])])] = 8
    dictionary[tuple([2, tuple([2, 1, 1, 1, 1, 1, 1])])] = 13
    dictionary[tuple([1, tuple([2, 2, 1, 1, 1, 1])])] = 5
    dictionary[tuple([2, tuple([2, 2, 1, 1, 1, 1])])] = 7
    dictionary[tuple([1, tuple([2, 2, 2, 1, 1])])] = 23
    dictionary[tuple([2, tuple([2, 2, 2, 1, 1])])] = 78
    dictionary[tuple([2, tuple([2, 2, 2, 2])])] = 31

    # three and less
    dictionary[tuple([1, tuple([3, 1, 1, 1, 1, 1])])] = 5
    dictionary[tuple([3, tuple([3, 1, 1, 1, 1, 1])])] = 16
    dictionary[tuple([1, tuple([3, 2, 1, 1, 1])])] = 6
    dictionary[tuple([2, tuple([3, 2, 1, 1, 1])])] = 11
    dictionary[tuple([3, tuple([3, 2, 1, 1, 1])])] = 4
    dictionary[tuple([1, tuple([3, 2, 2, 1])])] = 3
    dictionary[tuple([2, tuple([3, 2, 2, 1])])] = 18
    dictionary[tuple([3, tuple([3, 2, 2, 1])])] = 15
    dictionary[tuple([1, tuple([3, 3, 1, 1])])] = 14
    dictionary[tuple([3, tuple([3, 3, 1, 1])])] = 5
    dictionary[tuple([2, tuple([3, 3, 2])])] = 12
    dictionary[tuple([3, tuple([3, 3, 2])])] = 7

    # four and less
    dictionary[tuple([1, tuple([4, 1, 1, 1, 1])])] = 4
    dictionary[tuple([4, tuple([4, 1, 1, 1, 1])])] = 53
    dictionary[tuple([1, tuple([4, 2, 1, 1])])] = 9
    dictionary[tuple([2, tuple([4, 2, 1, 1])])] = 17
    dictionary[tuple([4, tuple([4, 2, 1, 1])])] = 16
    dictionary[tuple([2, tuple([4, 2, 2])])] = 24
    dictionary[tuple([4, tuple([4, 2, 2])])] = 36
    dictionary[tuple([1, tuple([4, 3, 1])])] = 1
    dictionary[tuple([3, tuple([4, 3, 1])])] = 0
    dictionary[tuple([4, tuple([4, 3, 1])])] = 2
    dictionary[tuple([4, tuple([4, 4])])] = 77

    # five and less
    dictionary[tuple([1, tuple([5, 1, 1, 1])])] = 3
    dictionary[tuple([5, tuple([5, 1, 1, 1])])] = 6
    dictionary[tuple([1, tuple([5, 2, 1])])] = 0
    dictionary[tuple([2, tuple([5, 2, 1])])] = 13
    dictionary[tuple([5, tuple([5, 2, 1])])] = 4
    dictionary[tuple([3, tuple([5, 3])])] = 50
    dictionary[tuple([5, tuple([5, 3])])] = 9

    # six, seven, eight and less
    dictionary[tuple([1, tuple([6, 1, 1])])] = 2
    dictionary[tuple([6, tuple([6, 1, 1])])] = 63
    dictionary[tuple([2, tuple([6, 2])])] = 5
    dictionary[tuple([6, tuple([6, 2])])] = 21
    dictionary[tuple([1, tuple([7, 1])])] = 1
    dictionary[tuple([7, tuple([7, 1])])] = 81
    dictionary[tuple([8, tuple([8])])] = 100

    """ N = 5
    dictionary = dict()

    # One and two
    dictionary[tuple([1, tuple([1, 1, 1, 1, 1])])] = 0
    dictionary[tuple([1, tuple([2, 1, 1, 1])])] = 1
    dictionary[tuple([2, tuple([2, 1, 1, 1])])] = 6
    dictionary[tuple([1, tuple([2, 2, 1])])] = 2
    dictionary[tuple([2, tuple([2, 2, 1])])] = 7

    # Three and less
    dictionary[tuple([1, tuple([3, 1, 1])])] = 3
    dictionary[tuple([3, tuple([3, 1, 1])])] = 11
    dictionary[tuple([2, tuple([3, 2])])] = 4
    dictionary[tuple([3, tuple([3, 2])])] = 15

    # Four and less, five.
    dictionary[tuple([1, tuple([4, 1])])] = 5
    dictionary[tuple([4, tuple([4, 1])])] = 70
    dictionary[tuple([5, tuple([5])])] = 80 """

    """ N = 4
    dictionary = dict()

    # One and two
    dictionary[tuple([1, tuple([1, 1, 1, 1])])] = 2
    dictionary[tuple([1, tuple([2, 1, 1])])] = 1
    dictionary[tuple([2, tuple([2, 1, 1])])] = 6
    dictionary[tuple([2, tuple([2, 2])])] = 4

    # Three and four
    dictionary[tuple([1, tuple([3, 1])])] = 3
    dictionary[tuple([3, tuple([3, 1])])] = 9
    dictionary[tuple([4, tuple([4])])] = 16 """


    print(dictToMatUsingLeastSquares(dictionary, N))
    print(dictToMat(dictionary, N))
