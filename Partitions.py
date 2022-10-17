# %%
import numpy as np
import itertools as it

# %%
"""
This cell implements the creation of various matrices A
for testing purpose.
"""
def makingOfA(N):
    At = np.random.randint(-10,10,(1,N))
    A = At.T.dot(At)
    return A

# A few test matrix with N=5
def testA(N):
    assert N == 5, "Must be a game with 5 players."
    A = np.array([
      [25, 20, 15, 10, 5],
      [24, 19, 14,  9, 4],
      [23, 18, 13,  8, 3],
      [22, 17, 12,  7, 2],
      [21, 16, 11,  6, 1]
    ])
    return A

def testATriangSuppWithZeros(N):
    assert N == 5, "Must be a game with 5 players."
    A = np.array([
      [ 89, -87, -82, -52, -81],
      [-96,  18, -70, 7, 0],
      [ 62,  71,  61, 0, 0],
      [-24, -16,   0, 0, 0],
      [ 15,   0,   0, 0, 0],
    ])
    return A

def testATriangSuppWithoutZeros(N):
    assert N == 5, "Must be a game with 5 players."
    A = np.array([
      [ 89, -87, -82, -52, -81],
      [-96,  18, -70,   7,   4],
      [ 62,  71,  61,  46,   6],
      [-24, -16,  78,  23, -80],
      [ 15,   2,  54,   6, -15],
    ])
    return A

def testAPositiveExternalities(N):
    assert N == 5, "Must be a game with 5 players."
    A = np.array([
      [ 20, 40, 60, 80, 100],
      [ 15, 20, 25, 30,   0],
      [  5, 10, 15,  0,   0],
      [  2,  5,  0,  0,   0],
      [  1,  0,  0,  0,   0],
    ])
    return A


def testANegativeExternalities(N):
    assert N == 5, "Must be a game with 5 players."
    A = np.array([
      [ 6, 5, 4, 3, 2],
      [ 7, 6, 5, 4, 0],
      [ 8, 7, 6, 0, 0],
      [ 9, 8, 0, 0, 0],
      [10, 0, 0, 0, 0],
    ])
    return A

lb_A = -100
ub_A = 100
def diagonalA(N):
    A = np.diag(np.random.randint(lb_A, ub_A, N))
    return A

def identityA(N):
    A = np.diag(np.ones(N))
    return A

def triangularLowerA(N):
    A = np.tril(np.random.randint(lb_A, ub_A, (N,N)))
    return A

def mirrorTriangularLowerA(N):
    B = np.tril(np.random.randint(lb_A, ub_A, (N,N)))
    A = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            A[i,j] = B[i, N-j-1]
    return A

def triangularUpperA(N):
    A = np.triu(np.random.randint(lb_A, ub_A, (N,N)))
    return A

def mirrorTriangularUpperA(N):
    B = np.triu(np.random.randint(lb_A, ub_A, (N,N)))
    A = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            A[i,j] = B[i, N-j-1]
    return A

def symmetricA(N):
    A = np.random.randint(lb_A, ub_A, (N,N))
    for i in range(N):
        for j in range(i+1, N):
            A[i,j] = A[j,i]
    return A

def symmetricANullDiago(N):
    A = np.random.randint(lb_A, ub_A, (N,N))
    for i in range(N):
        A[i,i] = 0
        for j in range(i+1, N):
            A[i,j] = A[j,i]
    return A

def antisymmetricA(N):
    A = np.random.randint(lb_A, ub_A, (N,N))
    for i in range(N):
        A[i,i] = 0
        for j in range(i+1, N):
            A[i,j] = -A[j,i]
    return A

def identicalRowsA(N):
    A = np.random.randint(lb_A, ub_A, (N,N))
    for i in range(1, N):
        for j in range(N):
            A[i,j] = A[0,j]
    return A

def identicalColumnsA(N):
    A = np.random.randint(lb_A, ub_A, (N,N))
    for i in range(1, N):
        for j in range(N):
            A[j,i] = A[j,0]
    return A

def identicalElementsA(N):
    A = np.random.randint(lb_A, ub_A) * np.ones((N,N))
    return A

# %%
def listCopy(l):
    new = []
    for i in l:
        if (type(i)==list):
            new.append(listCopy(i))
        else:
            new.append(i)
    return new

def partitions(a):
    parts = []
    for i in a:
        if len(parts) == 0:
            parts.append([[i]])
        else:
            temp = []
            for p1 in parts:
                for p2 in p1:
                    t1 = listCopy(p1)
                    for p3 in t1:
                        if p3==p2:
                            p3.append(i)
                            break
                    temp.append(t1)
                t1 = listCopy(p1)
                t1.append([i])
                temp.append(t1)
            parts = temp
    return parts
# %%
def list2Mat(partition):
    players = [i for coals in partition for i in coals]
    n = len(players)
    mat = np.zeros((n,n),dtype=int)
    for i, coal in enumerate(partition):
        for j in (coal):
            mat[i,j] = 1
    return mat.T

def Mat2list(mat):
    partition = []
    n = mat.shape[1]
    for j in range(n):
        coal = []
        for i in range(n):
            if mat[i,j] > 0.5:
                coal.append(i)
        if len(coal) > 0.5:
            partition.append(coal)
    return partition

# %%
def subcoalitionsForF2(a):
    """
    Given a coalition in a list form, return a list of all the ways
    the coalition could be split into two subcoalitions.
    """
    parts = []
    for i in a:
        if len(parts) == 0:
            parts.append([[i]])
        else:
            temp = []
            for p1 in parts:
                t1 = listCopy(p1)
                t1[0].append(i)
                t2 = listCopy(p1)
                if len(p1) < 2:
                    t2.append([i])
                else:
                    t2[1].append(i)
                temp.append(t1)
                temp.append(t2)
            parts = temp
    #print("subcoals", parts[1:])
    return parts[1:]

def neighboursForF2(a):
    """
    Given a partition in a list form, return a list of all the 
    partitions neighbour of that partition according to F2 (split).
    """
    neighbours = []
    for i, coal in enumerate(a):
        if len(coal) > 1:
            subcoals = subcoalitionsForF2(coal)
            for subcoal in subcoals:
                t1 = listCopy(a)
                t1.pop(i)
                t1.insert(i, subcoal[0])
                if len(subcoal) > 1:
                    t1.insert(i+1, subcoal[1])
                t1.sort()
                neighbours.append(t1)
    return neighbours

def neighboursForF4(a):
  """
  Given a partition in a list form, return a list of all the 
  partitions neighbour of that partition according to F4 (joining).
  """
  neighbours = []
  for i, coal1 in enumerate(a[:-1]):
      for j, coal2 in enumerate(a[i+1:]):
          #print("Coals", i, coal1, j+i+1, coal2)
          t1 = listCopy(a)
          t1.pop(j+i+1)
          t1[i] += coal2
          t1[i].sort()
          t1.sort()
          neighbours.append(t1)
  return neighbours

def compareCoals(a, b):
    """
    Given two partitions that are neighbours, find the indexes in the first
    partition of the changing coalitions.
    """
    diffs = []
    for i, coal in enumerate(a):
        if coal not in b:
            diffs.append(i)
    return diffs

def subcoalitionsForCore(coal, n):
    """
    Given a coalition in a list form, and the number of parting players from that coal,
    return all the resulting partitions of that coal.
    """
    parts = []
    combinations = it.combinations(coal, n)
    for combination in iter(combinations):
        temp = []
        temp.append(list(combination))
        temp.append([x for x in coal if x not in combination])
        parts.append((list(combination), temp))
    return parts

def neighboursForCore(a, n):
    """
    Given a partition in a list form, return a list of all the 
    partitions neighbour of that partition if there is only one subcoal of
    n players splitting from one coal in the original partition.
    Returns the index of the new coalition before the new partition, in a tuple.
    """
    neighbours = []
    if n <= 0: return neighbours
    for i, coal in enumerate(a):
        if len(coal) > n:
            subcoals = subcoalitionsForCore(coal, n)
            for subcoal in subcoals:
                t1 = listCopy(a)
                t1.pop(i)
                t1.insert(i, subcoal[1][0])
                if len(subcoal) > 1:
                    t1.insert(i+1, subcoal[1][1])
                t1.sort()
                t1.sort(key = len, reverse=True)
                neighbours.append((t1.index(subcoal[0]), t1))
        elif len(coal) == n:
            t1 = listCopy(a)
            neighbours.append((t1.index(coal), t1))
    return neighbours

# %%
if __name__ == "__main__":
    N = 5
    ps = partitions([i for i in range(N)])
    print(len(ps))
    A = np.random.randint(1, 10, (N,N))
    # v = np.ones((N,1))
    v = np.random.randint(1,10,(N,1))
    print('A: ',A)
    print('v: ',v)
    testPns = [12,32,5,16,25,9]
    testPartitions = [ps[i] for i in testPns]
    for partition in testPartitions:
        mat = list2Mat(partition)
        # val = v.T.dot(A.dot(mat))
        val = v.T.dot(mat)
        print('partition: ',partition)
        # print(A.dot(mat))
        print(val)
        print()
        print()
    
    for partition in testPartitions:
        print("\nPartition", partition)
        neighbours2 = neighboursForF2(partition)
        print("Neighbours F2", neighbours2)
        neighbours4 = neighboursForF4(partition)
        print("Neighbours F4", neighbours4)
        print("Partition : ", partition, "\n")
        neighboursCore = neighboursForCore(partition, 0)
        print("Neighbours Core 0", neighboursCore)
        neighboursCore = neighboursForCore(partition, 1)
        print("Neighbours Core 1", neighboursCore)
        neighboursCore = neighboursForCore(partition, 2)
        print("Neighbours Core 2", neighboursCore)
        neighboursCore = neighboursForCore(partition, 3)
        print("Neighbours Core 3", neighboursCore)
        neighboursCore = neighboursForCore(partition, 4)
        print("Neighbours Core 4", neighboursCore)
        neighboursCore = neighboursForCore(partition, 5)
        print("Neighbours Core 5", neighboursCore)



    # 
    # for i in testPns:
    #     pp = ps[i]
    #     print(pp)
    #     m = list2Mat(pp)
    #     print(m)
    #     print(Mat2list(m))
    #     print()

def partitionsK(a, k=10):
    parts = []
    for i in a:
        if len(parts) == 0:
            parts.append([[i]])
        else:
            temp = []
            for p1 in parts:
                for p2 in p1:
                    t1 = listCopy(p1)
                    for p3 in t1:
                        if p3==p2:
                            if len(p3)<k:
                                p3.append(i)
                            break
                    temp.append(t1)
                t1 = listCopy(p1)
                t1.append([i])
                temp.append(t1)
            parts = temp
    parts2 = []
    for pp in parts:
        t2 = [j for k in pp for j in k]
        if len(t2)==len(a):
            parts2.append(pp)
    return parts2
