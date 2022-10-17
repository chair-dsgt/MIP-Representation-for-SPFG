import numpy as np

def listCopy(l):
    new = []
    for i in l:
        if (type(i)==list):
            new.append(listCopy(i))
        else:
            new.append(i)
    return new

def getVerdict(Qmat, n, h):
    """
    Function that verifies if the rank of the Q matrix equals to h+1 or not.
    """
    a = np.array(Qmat)
    rank_a = np.linalg.matrix_rank(a)
    verdict = ""
    if n-h < n/2 and h+1 > rank_a and h > 4:
      verdict = "The rank of matrix Q for n = " + str(n) + " is smaller than h+1 = " + str(h+1) + " for column number " + str(n-h) + ". It is = " + str(rank_a) + "."
      print(verdict)
    elif n-h >= n/2 and h != rank_a and h > 4:
      verdict = "The rank of matrix Q for n = " + str(n) + " is different from h = " + str(h) + " for column number " + str(n-h) + ". It is = " + str(rank_a) + "."
      print(verdict)

def addNewPartitionToPartList(templst, col, app1, app2):
    templst[0] -= 1
    templst[col] += 1
    if templst not in app1:
        app1.append(templst)
        app2.append(templst)

def buildPartitionsForN(n):
    """
    Function that returns a 2D list whose rows are partitions of a game with n players.
    """
    all_part = []
    queue = []

    # first partition
    a = [n if i == 0 else 0 for i in range(n)]
    addNewPartitionToPartList(a, 0, all_part, queue)

    while len(queue) != 0:
        parent = queue.pop(0)
        if parent[0] != 1:
            for col in range(1, n):
                if parent[0] > parent[1]:
                    if parent[col] == 0:
                        # We append to the end and break loop (remaining col will also be 0)
                        addNewPartitionToPartList(listCopy(parent), col, all_part, queue)
                        break
                    elif (col == 1 and parent[col-1] >= parent[col] + 2) or (col > 1 and parent[col-1] >= parent[col] + 1):
                        # This coal has space to grow
                        addNewPartitionToPartList(listCopy(parent), col, all_part, queue)
    return all_part

def constructQForH(part_list, n, h):
    """
    Function that receives a partition list and selects all the partitions from it that contains at least one n-h element.
    """
    Qmat = []
    for vect in part_list:
        if n-h in vect:
            Qmat.append(vect)
    return Qmat

def RankStudy(n):
    """
    Function that study the rank of all Q matrices for games of n players.
    """
    partitions = buildPartitionsForN(n)
    for h in range(n):
        Qmat = constructQForH(partitions, n, h)
        getVerdict(Qmat, n, h)

if __name__ == "__main__":
    """
    Runs the loops for multiple n, allows to see that there is a single exception to the rules.
    Rules :
    1) If n-h >= n/2, then we have rank=h. In other words, those are the columns where the studied coal is big.
    2) If n-h < n/2 and n >= 6, h >= 5, then we have rank=h+1. In other words, those are the columns where the studied coal is small.
    Exception : n=7, h=5. This combo has rank=h, even if we are studying the « small » coal of 2.
    h is a variable representing the number of players that are not in the studied coal of a given column. 
    """
    for n in range(3,20):
        RankStudy(n)
