# %%
import gurobipy as gp
import math
import numpy as np
import Partitions as pt

# %%
N = 5
s = np.array([pow(2,(N-1-i)) for i in range(N)])
nsuite = np.array([i for i in range(N+1)])
A = pt.mirrorTriangularUpperA(N)
B = np.append(np.zeros((N, 1)), A, axis=1)

def setParams(myA):
    """
    If another file uses this one, we can set params here before running the SWPlusChecksLoop.
    """
    assert myA.shape[0] == myA.shape[1], "The matrix A must be square."
    global N
    N = myA.shape[0]
    global A
    A = myA
    global s
    s = np.array([pow(2,(N-1-i)) for i in range(N)])
    global nsuite
    nsuite = np.array([i for i in range(N+1)])
    global B
    B = np.append(np.zeros((N, 1)), A, axis=1)

# %%

def payoff2(B, partnMat):
    """
    We define a partition function form game, that is defined by a matrix 
    A of size NxN, that will be able to render symmetric game.
    A coalition of size k will receive its payoff from the kth column in A (augmented with a 0 column at first).  
    Given a partition matrix partnMat, and the partition-function-defining A, 
    this function returns the payoff vector corresponding to the partition partnMat.
    """
    p2t = partnMat.sum(axis=0)
    uT2 = np.array([p2t.dot(B[0:N,p2t[j]]) for j in range(N)]).flatten()
    return uT2

# %%
def F1Check(B, partnMat):
    """
    Given a PFG defining B and 
    a partition matrix partnMat we wish to evaluate,
    Check if partnMat respect the F1 criterion.
    F1 : Total partition payoff is bigger or equal to p0's.
    """
    u1 = payoff2(B, pt.list2Mat([[i] for i in range(N)]))
    u2 = payoff2(B, partnMat)
    return True if sum(u2) >= sum(u1) else False

def F3Check(B, partnMat):
    """
    Given a PFG defining B and 
    a partition matrix partnMat we wish to evaluate,
    Check if partnMat respect the F3 criterion.
    F3 : Per coal payoff is bigger or equal to p0 simili-coal's.
    """
    u1 = payoff2(B, pt.list2Mat([[i] for i in range(N)]))
    u2 = payoff2(B, partnMat)
    ppt = pt.Mat2list(partnMat)
    isBigger = True
    for j in range(len(ppt)):
        if sum([u1[i] for i in ppt[j]]) > u2[j]:
            isBigger = False
            break
    return isBigger

def F2Check(B, partnMat):
    """
    Given a PFG defining B and 
    a partition matrix partnMat we wish to evaluate,
    Check if partnMat respect the F2 criterion.
    F2 : Per coal payoff is bigger or equal to all splitting into 2.
    """
    u2 = payoff2(B, partnMat)
    original = pt.Mat2list(partnMat)
    isBigger = True
    for neighbour in pt.neighboursForF2(original):
        u1 = payoff2(B, pt.list2Mat(neighbour))
        if sum([u1[i] for i in pt.compareCoals(neighbour, original)]) > sum([u2[i] for i in pt.compareCoals(original, neighbour)]):
            isBigger = False
            break
    return isBigger
    
def F4Check(B, partnMat):
    """
    Given a PFG defining B and 
    a partition matrix partnMat we wish to evaluate,
    Check if partnMat respect the F4 criterion.
    F4 : Per coal payoff is bigger or equal to all joining of 2.
    """
    u2 = payoff2(B, partnMat)
    original = pt.Mat2list(partnMat)
    isBigger = True
    for neighbour in pt.neighboursForF4(original):
        u1 = payoff2(B, pt.list2Mat(neighbour))
        if sum([u1[i] for i in pt.compareCoals(neighbour, original)]) > sum([u2[i] for i in pt.compareCoals(original, neighbour)]):
            isBigger = False
            break
    return isBigger

# %%
def addF1ConstraintToModel(B, M, u):
    return M.addConstr(gp.quicksum(u[i] for i in range(N)) >= N*sum(B[i,1] for i in range(N)),name="f1")

def addF3ConstraintToModel(B, M, X, u):
    return M.addConstrs((u[i] >= gp.quicksum(X[j,i] *sum(B[k,1] for k in range(N)) for j in range(N)) for i in range(N)),name="f3")

def addF2ConstraintToModel(B, M, xMat, u):
    constr_lst = []
    original = pt.Mat2list(xMat)
    for k, neighbour in enumerate(pt.neighboursForF2(original)):
        u1 = payoff2(B, pt.list2Mat(neighbour))
        constr_lst.append(M.addConstr(sum([u[i] for i in pt.compareCoals(original, neighbour)]) >= sum([u1[i] for i in pt.compareCoals(neighbour, original)]),name="f2_"+str(k)))
    return constr_lst

def addF4ConstraintToModel(B, M, xMat, u):
    constr_lst = []
    original = pt.Mat2list(xMat)
    for k, neighbour in enumerate(pt.neighboursForF4(original)):
        u1 = payoff2(B, pt.list2Mat(neighbour))
        constr_lst.append(M.addConstr(sum([u[i] for i in pt.compareCoals(original, neighbour)]) >= sum([u1[i] for i in pt.compareCoals(neighbour, original)]),name="f4_"+str(k)))
    return constr_lst

def addBanConstraintToModel(M, xMat, X, n_ite):
    """
    Constraint that forbids var X to take the value of xMat for the next optim.
    """
    print("BAN")
    return M.addConstr(sum((X[j,i] * xMat[j,i] + (1 - X[j,i]) * (1 - xMat[j,i])) for i in range(N) for j in range(N)) <= N*N - 0.00001, name="ban_"+str(n_ite))

# %%

def findReasonablePayoffSplitting(B, solMat, max_tot_val, u):
    """
    Given a PFG defining B and a solution for the coalition formation, find a payoff splitting.
    """
    # Little model to put constraints on y (we have a basic sum to respect, but otherwise...)
    N = B.shape[0]
    assert B.shape[1] == N+1, "B must be square +1 on column count"
    M = gp.Model("payoffSplit")
    y = M.addVars(N,vtype=gp.GRB.CONTINUOUS,name="y",lb=-gp.GRB.INFINITY)
    C_ysum = M.addConstr(gp.quicksum(y[i] for i in range(N)) <= max_tot_val,name="y_sum")
    M.setObjective(gp.quicksum(y[i] for i in range(N)),gp.GRB.MAXIMIZE)
    M.params.nonconvex=2
    M.Params.OutputFlag = 0

    # Looping to gradually add core constraints based on how many players are entering an additional coal (from the same original coal).
    # Right now, there is a limit of one original coal and one new destination coal for defining the neighbourhood.
    # When n = 1, this correspond to a Nash situation.
    for n in range(1,N):
        level_constr = []
        for i, neighbour in enumerate(pt.neighboursForCore(pt.Mat2list(solMat), n)):
            #print(payoff2(B, pt.list2Mat(neighbour[1])))
            level_constr.append(
                M.addConstr(gp.quicksum(y[j] for j in neighbour[1][neighbour[0]]) >= payoff2(B, pt.list2Mat(neighbour[1]))[neighbour[0]],
                name="y_neigh_" + str(n) + "_" + str(i))
            )
        M.optimize()
        if (M.SolCount == 0):
            print("No more solutions at level", n, "backtrack model to last level.")
            M.remove(level_constr)
        else:
            print("Found solution at level", n, ":", [ b.X for b in y.values()])

    # To test if there are extracoal exchanges. If this is added and the model is not feasible anymore, it means there are.
    C_ydef = []
    for j, v in enumerate(u.values()):
        C_ydef.append(M.addConstr((gp.quicksum(solMat[i,j]*y[i] for i in range(N)) >= v.X),name="y_def_" + str(j)))
    M.optimize()
    if (M.SolCount == 0):
        print("No more solutions, means there are extracoal exchanges.")
    else:
        print("After extracoal exchanges check :", [ b.X for b in y.values()])

# %%

def makeGurobiModelBiggestCoalFirst(B):
    """
    Given a PFG defining B, make a gurobi model that identifies the
    partition with highest total payoff. In the matrix X, instead of having the first player
    come first in the coalitions, the coalitions will be ordered by the most numerous to the least.
    The player order will serve as a secondary criterion.
    """
    N = B.shape[0]
    assert B.shape[1] == N+1, "B must be square +1 on column count"
    M = gp.Model("PFG")
    X = M.addVars(N,N,vtype=gp.GRB.BINARY,name="X")
    q = M.addVars(N-1,vtype=gp.GRB.BINARY,name="q")
    u = M.addVars(N,vtype=gp.GRB.CONTINUOUS,name="u",lb=-gp.GRB.INFINITY)
    # In ssnjpc, values are scores given index is number of players in the coal.
    ssnjpc = M.addVars(N+1,vtype=gp.GRB.CONTINUOUS,name="ssnjpc",lb=-gp.GRB.INFINITY)
    # In njpcbin, rows are coals, columns are index, there are ones where in that coal, there are index as number of players.
    njpcbin  = M.addVars(N,N+1,vtype=gp.GRB.BINARY,name="njpcbin")
    y = M.addVars(N,vtype=gp.GRB.CONTINUOUS,name="y",lb=-gp.GRB.INFINITY)

    # Constraint for X matrix to be clean and allow a single configuration for every partition.
    C_xsum = M.addConstrs((X.sum(i,'*') == 1 for i in range(N)),name="X_sum")
    C_symBr = M.addConstrs(
        (X.sum('*',j) >= X.sum('*',j+1) for j in range(N-1)), name = "symmBreak"
        )
    C_symBr2 = []
    for j in range(N-1):
        q1 = M.addGenConstrIndicator(q[j], 0, X.sum('*',j) - X.sum('*',j+1) >= 1)
        q2 = M.addGenConstrIndicator(q[j], 1, X.sum('*',j) - X.sum('*',j+1) == 0)
        q3 = M.addGenConstrIndicator(q[j], 1, gp.quicksum(s[i]*X[i,j] for i in range(N)) >= gp.quicksum(s[i]*X[i,j+1] for i in range(N)))
        C_symBr2.append((q1, q2, q3))

    # Constraints for the computation of the payoff
    C_ssnjpc = M.addConstrs((ssnjpc[j] == gp.quicksum(X.sum('*',i)*B[i, j] for i in range(N)) for j in range(N+1)),name="ssnjpc")
    C_njpcbinsum = M.addConstrs((njpcbin.sum(j,'*') == 1 for j in range(N)),name="njpcbin_sum")
    C_njpcbinone = M.addConstrs((X.sum('*',i) == gp.quicksum(njpcbin[i, j]*nsuite[j] for j in range(N+1)) for i in range(N)),name="njpcbin_one")
    C_u = M.addConstrs((u[i] == gp.quicksum(njpcbin[i, j]*ssnjpc[j] for j in range(N+1)) for i in range(N)),name="u")
    C_ydef = M.addConstrs(
        (gp.quicksum(X[i,j]*y[i] for i in range(N)) == u[j] for j in range(N)),name="y_def"
    )

    M.setObjective(gp.quicksum(u[i] for i in range(N)),gp.GRB.MAXIMIZE)
    M.params.nonconvex=2
    return M, {'X':X, 'q':q, 'u':u, 'y':y, 'ssnjpc':ssnjpc, 'njpcbin':njpcbin,
                'C_xsum':C_xsum, 'C_symBr':C_symBr, 'C_symBr2':C_symBr2, 'C_ssnjpc':C_ssnjpc,
                'C_njpcbinsum':C_njpcbinsum, 'C_njpcbinone':C_njpcbinone, 'C_u':C_u, 'C_ydef':C_ydef}

# %%
def TestAllFChecks():
    """
    Checking the F's
    F1 & F3 : The partition's coalitions (and the partition itself) are making more than their equivalent in p_0
    F2 & F4 : Find the neighborhood; Drag conclusions on the current partition, and maybe also on the neighbors
    (but will have to stock).
    Right now, everything is represented by check functions, to then verify if an X returned by Gurobi is
    satisfying, but they could also be declined into constraints to add/remove to the model (by example, to see if
    a criteria leaves much of the space or not).
    """
    print("F1 on p0", F1Check(B, pt.list2Mat([[i] for i in range(N)])))
    print("F2 on p0", F2Check(B, pt.list2Mat([[i] for i in range(N)])))
    print("F3 on p0", F3Check(B, pt.list2Mat([[i] for i in range(N)])))
    print("F4 on p0", F4Check(B, pt.list2Mat([[i] for i in range(N)])))

    for ppt in [[[0,1,2,3,4]], [[0,3],[1,2],[4]], [[0],[1,2,3],[4]], [[0,4],[1,3],[2]]]:
        print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
        print("F1", F1Check(B, pt.list2Mat(ppt)))
        print("F2", F2Check(B, pt.list2Mat(ppt)))
        print("F3", F3Check(B, pt.list2Mat(ppt)))
        print("F4", F4Check(B, pt.list2Mat(ppt)))

# %%
def PrintAPartitionSol(B, objVal, optPartn, values_u, values_y, values_q, values_ssnjpc, values_njpcbin, solNumber=0):  
    print("#############", solNumber, "#############")
    print("Optimal partition :", optPartn)
    print("Values for u :", values_u)
    print("Values for y :", values_y)
    print("Values for q :", values_q)
    print("Values for ssnjpc :", values_ssnjpc)
    print("Values for njpcbin :", values_njpcbin)
    print("Objective val :", objVal)
    print("F1", F1Check(B, pt.list2Mat(optPartn)))
    print("F2", F2Check(B, pt.list2Mat(optPartn)))
    print("F3", F3Check(B, pt.list2Mat(optPartn)))
    print("F4", F4Check(B, pt.list2Mat(optPartn)))

def PrintAllSols(B, M):
    # Print different sol loop
    for sol in range(0, M.SolCount):
        M.setParam(gp.GRB.Param.SolutionNumber, sol)
        values_X = np.reshape(M.Xn[0:N*N], (N,N))
        values_q = np.reshape(M.Xn[N*N:N*N+N-1], (1, N-1))
        values_u = np.reshape(M.Xn[N*N+N-1:N*N+N*2-1], (1, N))
        values_ssnjpc = np.reshape(M.Xn[N*N+N*2-1:N*N+N*3], (1, N+1))
        values_njpcbin = np.reshape(M.Xn[N*N+N*3:N*(2*N+4)], (N, N+1))
        values_y = np.reshape(M.Xn[N*(2*N+4):N*(2*N+5)], (1, N))
        optPartn = pt.Mat2list(values_X)
        PrintAPartitionSol(B, M.PoolObjVal, optPartn, values_u, values_y, values_q, values_ssnjpc, values_njpcbin, sol)

def PrintOneSol(B, M, n_ite):
    # Print one sol
    values_X = np.reshape(M.Xn[0:N*N], (N,N))
    values_q = np.reshape(M.Xn[N*N:N*N+N-1], (1, N-1))
    values_u = np.reshape(M.Xn[N*N+N-1:N*N+N*2-1], (1, N))
    values_ssnjpc = np.reshape(M.Xn[N*N+N*2-1:N*N+N*3], (1, N+1))
    values_njpcbin = np.reshape(M.Xn[N*N+N*3:N*(2*N+4)], (N, N+1))
    values_y = np.reshape(M.Xn[N*(2*N+4):N*(2*N+5)], (1, N))
    optPartn = pt.Mat2list(values_X)
    PrintAPartitionSol(B, M.PoolObjVal, optPartn, values_u, values_y, values_q, values_ssnjpc, values_njpcbin, n_ite)
    return optPartn

def SWPlusChecksLoop():
    """
    This loop finds the SW-attractive partition according to the model, and then verifies if it passes the
    F_1 - F_4 checks. If it doesn't, this partition is then banned from the model, and we go to the next iteration.
    """
    print(A)
    # This search mode (1) makes gurobi retain PoolSolutions of the best solutions that got evaluated during
    # the search for the optimal solution.
    # The search mode (2) would make gurobi retain PoolSolutions of the best solutions overall (longer).
    gp.GRB.PoolSearchMode=2
    gp.GRB.PoolSolutions=5
    M, detDict = makeGurobiModelBiggestCoalFirst(B)
    X = detDict['X']
    u = detDict['u']
    # This will be used to retrieve the optimal partition at the end.
    optmlPrttn = None
    
    M.Params.OutputFlag = 0
    n_ite = 0
    # LOOP (while we haven't found anything promising or tested enough solutions)
    while n_ite < N*N:
        # Optimize and keep the solution
        M.optimize()
        if (M.SolCount == 0):
            print("SolCount is 0 after first optimization of ite", n_ite)
            break
        else:
            optmlPrttn = PrintOneSol(B, M, n_ite)
        opt_value = M.PoolObjVal
        xMat = np.zeros((N,N), dtype=int)
        for i in range(N):
            for j in range(N):
                if X[i,j].X > 0.5:
                    xMat[i,j] = 1
                else:
                    xMat[i,j] = 0
        # Verify the checks
        check1_result = F1Check(B, xMat)
        check2_result = F2Check(B, xMat)
        check3_result = F3Check(B, xMat)
        check4_result = F4Check(B, xMat)
        # If the checks are passing, the solution is really optimal.
        if (check1_result and check2_result and check3_result and check4_result):
            print("The solution is passing all stability checks.")
            findReasonablePayoffSplitting(B, xMat, opt_value, u)
            break
        # Else it's not, ban the solution!
        else:
            print("The solution is failing some/all checks.")
            addBanConstraintToModel(M, xMat, X, n_ite)
            n_ite += 1
    return optmlPrttn

def TestFamilyOfRepresentedModels():
    """
    So we can see what we can get with different matrices A.
    """

    print("\nFive")
    ppt = [[0,1,2,3,4]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))

    print("\nFour-one")
    ppt = [[0,1,2,3],[4]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[0,1,2,4],[3]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[0,1,3,4],[2]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[0,2,3,4],[1]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[1,2,3,4],[0]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))

    print("\nThree-two")
    ppt = [[0,1,2],[3,4]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[0,1,3],[2,4]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[0,2,3],[1,4]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[1,2,3],[0,4]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[0,1,4],[2,3]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[0,2,4],[1,3]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[1,2,4],[0,3]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[0,3,4],[1,2]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[1,3,4],[0,2]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[2,3,4],[0,1]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))

    print("\nThree-one-one")
    ppt = [[0,1,2],[3],[4]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[0,1,3],[2],[4]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[0,2,3],[1],[4]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[1,2,3],[0],[4]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[0,1,4],[2],[3]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[0,2,4],[1],[3]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[1,2,4],[0],[3]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[0,3,4],[1],[2]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[1,3,4],[0],[2]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[2,3,4],[0],[1]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))

    print("\nTwo-two-one")
    ppt = [[0,1],[2,3],[4]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt =[[0,2],[1,3],[4]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt =[[0,3],[1,2],[4]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt =[[0,1],[2,4],[3]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt =[[0,2],[1,4],[3]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt =[[0,4],[1,2],[3]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt =[[0,1],[3,4],[2]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt =[[0,3],[1,4],[2]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt =[[0,4],[1,3],[2]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt =[[0,2],[3,4],[1]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt =[[0,3],[2,4],[1]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt =[[0,4],[2,3],[1]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt =[[1,2],[3,4],[0]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt =[[1,3],[2,4],[0]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt =[[1,4],[2,3],[0]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))

    print("\nTwo-one-one-one")
    ppt =[[0,1],[2],[3],[4]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt = [[0,2],[1],[3],[4]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt =[[0,3],[1],[2],[4]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt =[[0,4],[1],[2],[3]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt =[[1,2],[0],[3],[4]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt =[[1,3],[0],[2],[4]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt =[[1,4],[0],[2],[3]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt =[[2,3],[0],[1],[4]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt =[[2,4],[0],[1],[3]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))
    ppt =[[3,4],[0],[1],[2]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))

    print("\nOne-one-one-one-one")
    ppt =[[0],[1],[2],[3],[4]]
    print(ppt, payoff2(B, pt.list2Mat(ppt)), sum(payoff2(B, pt.list2Mat(ppt))))

# %%
if __name__ == "__main__":
    SWPlusChecksLoop()
    #TestFamilyOfRepresentedModels()
