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
""" A = np.array([
  [25, 20, 15, 10, 5],
  [24, 19, 14,  9, 4],
  [23, 18, 13,  8, 3],
  [22, 17, 12,  7, 2],
  [21, 16, 11,  6, 1]
]) """
# Next 2 matrices are proof that only numbers on left sup. triangle are important.
""" A = np.array([
  [ 89, -87, -82, -52, -81],
  [-96,  18, -70, 7, 0],
  [ 62,  71,  61, 0, 0],
  [-24, -16,   0, 0, 0],
  [ 15,   0,   0, 0, 0],
]) """
""" A = np.array([
  [ 89, -87, -82, -52, -81],
  [-96,  18, -70,   7,   4],
  [ 62,  71,  61,  46,   6],
  [-24, -16,  78,  23, -80],
  [ 15,   2,  54,   6, -15],
]) """
# Next 2 matrices are examples of positive/negative externalities
""" A = np.array([
  [ 20, 40, 60, 80, 100],
  [ 15, 20, 25, 30,   0],
  [  5, 10, 15,  0,   0],
  [  2,  5,  0,  0,   0],
  [  1,  0,  0,  0,   0],
]) """
""" A = np.array([
  [ 6, 5, 4, 3, 2],
  [ 7, 6, 5, 4, 0],
  [ 8, 7, 6, 0, 0],
  [ 9, 8, 0, 0, 0],
  [10, 0, 0, 0, 0],
]) """

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

def makeGurobiModelMcCormick(B):
    """
    Given a PFG defining B, make a gurobi model that identifies the
    partition with highest total payoff. In the matrix X, instead of having the first player
    come first in the coalitions, the coalitions will be ordered by the most numerous to the least.
    The player order will serve as a secondary criterion.
    Also, since we have a multiplication between two variables, we replace those with a McCormick envelope for convexification.
    """
    N = B.shape[0]
    assert B.shape[1] == N+1, "B must be square +1 on column count"
    M = gp.Model("PFG")
    X = M.addVars(N,N,vtype=gp.GRB.BINARY,name="X")
    q = M.addVars(N-1,vtype=gp.GRB.BINARY,name="q")
    u = M.addVars(N,vtype=gp.GRB.CONTINUOUS,name="u",lb=-gp.GRB.INFINITY)
    # In njpcbin, columns are coals, and there are ones where in that coal, there are index of row as number of players.
    njpcbin  = M.addVars(N+1,N,vtype=gp.GRB.BINARY,name="njpcbin")
    # In W, we have four dimensions NxNx(N+1)xN, and every element W_irkj represent the product between X_ir and njpcbin_kj.
    W = M.addVars(N,N,N+1,N,vtype=gp.GRB.BINARY,name="W")
    # y is not useful to the model, it's only there to give an idea of what the payoff splitting within the coalitions would give.
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
    
    # Constraints for matrix W
    C_W = []
    for i in range(N):
        for r in range(N):
            for k in range(N+1):
                for j in range(N):
                    w1 = M.addConstr(W[i,r,k,j] >= X[i,r] + njpcbin[k,j] -1,name="C_w1_"+str(i)+"_"+str(r)+"_"+str(k)+"_"+str(j))
                    w2 = M.addConstr(W[i,r,k,j] >= 0,name="C_w2_"+str(i)+"_"+str(r)+"_"+str(k)+"_"+str(j))
                    w3 = M.addConstr(W[i,r,k,j] <= X[i,r],name="C_w3_"+str(i)+"_"+str(r)+"_"+str(k)+"_"+str(j))
                    w4 = M.addConstr(W[i,r,k,j] <= njpcbin[k,j],name="C_w4_"+str(i)+"_"+str(r)+"_"+str(k)+"_"+str(j))
                    C_W.append((w1,w2,w3,w4))

    # Constraints for the computation of the payoff
    C_njpcbinsum = M.addConstrs((njpcbin.sum('*',j) == 1 for j in range(N)),name="njpcbin_sum")
    C_njpcbinone = M.addConstrs((X.sum('*',j) == gp.quicksum(nsuite[i]*njpcbin[i, j] for i in range(N+1)) for j in range(N)),name="njpcbin_one")
    C_u = M.addConstrs((u[j] == gp.quicksum(gp.quicksum(gp.quicksum(W[i,r,k,j]*B[r,k] for r in range(N)) for k in range(N+1)) for i in range(N)) for j in range(N)),name="u")
    C_ydef = M.addConstrs((gp.quicksum(X[i,j]*y[i] for i in range(N)) == u[j] for j in range(N)),name="y_def")

    M.setObjective(gp.quicksum(u[i] for i in range(N)),gp.GRB.MAXIMIZE)
    M.params.nonconvex=2
    return M, {'X':X, 'q':q, 'u':u, 'y':y, 'W':W, 'njpcbin':njpcbin,
                'C_xsum':C_xsum, 'C_symBr':C_symBr, 'C_symBr2':C_symBr2, 'C_W':C_W,
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
def PrintAPartitionSol(B, objVal, optPartn, values_u, values_y, values_q, values_w, values_njpcbin, solNumber=0):  
    print("#############", solNumber, "#############")
    print("Optimal partition :", optPartn)
    print("Values for u :", values_u)
    print("Values for y :", values_y)
    print("Values for q :", values_q)
    #print("Values for W :", values_w)
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
        values_W = np.reshape(M.Xn[N*N+N*2-1:(N*N+N*2-1)+(N*N*(N+1)*N)], (N,N,N+1,N))
        values_njpcbin = np.reshape(M.Xn[(N*N+N*2-1)+(N*N*(N+1)*N):(N*N+N*2-1)+(N*N*(N+1)*N)+(N+1)*N], (N, N+1))
        values_y = np.reshape(M.Xn[(N*N+N*2-1)+(N*N*(N+1)*N)+(N+1)*N:(N*N+N*2-1)+(N*N*(N+1)*N)+(N+1)*N+N], (1, N))
        optPartn = pt.Mat2list(values_X)
        PrintAPartitionSol(B, M.PoolObjVal, optPartn, values_u, values_y, values_q, values_W, values_njpcbin, sol)

def PrintOneSol(B, M, n_ite):
    # Print one sol
    values_X = np.reshape(M.Xn[0:N*N], (N,N))
    values_q = np.reshape(M.Xn[N*N:N*N+N-1], (1, N-1))
    values_u = np.reshape(M.Xn[N*N+N-1:N*N+N*2-1], (1, N))
    values_W = np.reshape(M.Xn[N*N+N*2-1:(N*N+N*2-1)+(N*N*(N+1)*N)], (N,N,N+1,N))
    values_njpcbin = np.reshape(M.Xn[(N*N+N*2-1)+(N*N*(N+1)*N):(N*N+N*2-1)+(N*N*(N+1)*N)+(N+1)*N], (N, N+1))
    values_y = np.reshape(M.Xn[(N*N+N*2-1)+(N*N*(N+1)*N)+(N+1)*N:(N*N+N*2-1)+(N*N*(N+1)*N)+(N+1)*N+N], (1, N))
    optPartn = pt.Mat2list(values_X)
    PrintAPartitionSol(B, M.PoolObjVal, optPartn, values_u, values_y, values_q, values_W, values_njpcbin, n_ite)

def SWPlusChecksLoop():
    """
    This loop finds the SW-attractive partition according to the model, and then verifies if it passes the
    F_1 - F_4 checks. If it doesn't, this partition is then banned from the model, and we go to the next iteration.
    """
    # This search mode (1) makes gurobi retain PoolSolutions of the best solutions that got evaluated during
    # the search for the optimal solution.
    # The search mode (2) would make gurobi retain PoolSolutions of the best solutions overall (longer).
    gp.GRB.PoolSearchMode=2
    gp.GRB.PoolSolutions=5
    M, detDict = makeGurobiModelMcCormick(B)
    X = detDict['X']
    u = detDict['u']
    
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
            PrintOneSol(B, M, n_ite)
        opt_value = M.PoolObjVal
        xMat = np.zeros((N,N), dtype=int)
        for i in range(N):
            for j in range(N):
                if X[i,j].X > 0.5:
                    xMat[i,j] = 1
                else:
                    xMat[i,j] = 0
        # Adding the constraints for f1 and f3 (that are consistent all throughout opt)
        f1_constr = addF1ConstraintToModel(B, M, u)
        f3_constr = addF3ConstraintToModel(B, M, X, u)
        # Generate f2_constr and f4_constr
        f2_constr = addF2ConstraintToModel(B, M, xMat, u)
        f4_constr = addF4ConstraintToModel(B, M, xMat, u)
        # Optimize again
        M.optimize()
        # If the solution has not changed, it is really optimal.
        if (M.Status == gp.GRB.OPTIMAL and M.SolCount > 0 and math.isclose(opt_value, M.PoolObjVal, rel_tol=1e-5)):
            print("The second opt value is equal to the first one.")
            yMat = np.zeros((N,N), dtype=int)
            solutionChanged=False
            for i in range(N):
                for j in range(N):
                    if X[i,j].X > 0.5:
                        yMat[i,j] = 1
                    else:
                        yMat[i,j] = 0
                    if xMat[i,j] != yMat[i,j]:
                        solutionChanged = True
            if not solutionChanged:
                break
            print("But the solution is not passing all checks.")
        # Else it's not, ban the solution, and remove f2_constr and f4_constr
        else:
            print("The second opt value is different from the first one.")
            if (M.SolCount == 0):
                print("Stop at second optim, no sol found.")
                # break
            addBanConstraintToModel(M, xMat, X, n_ite)
            M.remove(f1_constr)
            M.remove(f3_constr)
            M.remove(f2_constr)
            M.remove(f4_constr)
            n_ite += 1

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
SWPlusChecksLoop()
#TestFamilyOfRepresentedModels()
