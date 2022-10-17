# %%
import csv
import itertools
import math
import numpy as np
import Partitions as pt
import RankStudy as rs
import RankCheck as rc
import SymPartiFind as spf

# %%
def findNEPrices(coef, n=5, A=100.0, ALPHA=30.0):
    """
    findNEPrices
    Function that returns the prices for one player of all coalitions when they reach NE in the
    Bertrand problem.
    """
    sum_of_terms = sum(coef[i] / (2 + 2*ALPHA - (ALPHA/n)*coef[i]) for i in range(n))
    return [A / ((2 + 2*ALPHA - (ALPHA/n)*coef[i]) * (1 - (ALPHA/n)*sum_of_terms)) for i in range(n)]

def buildPFGDictBertrand(n=5, A=100.0, ALPHA=30.0):
    """
    buildPFGDictBertrand
    Function that returns the dictionary of expected NE profits for all embedded coalitions in the
    Bertrand problem.
    """
    dictionary = dict()
    partitions = rs.buildPartitionsForN(n)
    for partition in partitions:
        prices_for_part = findNEPrices(partition, n, A, ALPHA)
        total_price_sum = np.matmul(partition, prices_for_part)
        for i, coalition in enumerate(partition):
          if coalition != 0 and tuple([coalition, tuple(partition)]) not in dictionary:
              demand_i = A - (1 + ALPHA) * prices_for_part[i] + (ALPHA / n) * total_price_sum
              dictionary[tuple([coalition, tuple(partition)])] = prices_for_part[i] * demand_i * coalition
    return dictionary

def evaluateEpsilon(dict, matA):
    """
    evaluateEpsilon
    Function that finds, for each dict entry, the absolute difference between what the dict has
    and what matA returns. The epsilon will be the l-infinity norm (biggest difference of all). 
    """
    bigDiff = 0
    for key, value in dict.items():
        if key[0] != 0:
            valMatA = np.dot(key[1], matA[:, key[0]-1])
            if abs(valMatA - value) > bigDiff:
                bigDiff = abs(valMatA - value)
    if math.isclose(bigDiff, 0, rel_tol=1e-5):
        bigDiff = 0
    return bigDiff

def payoffArrayFromDict(part, dictionary, n):
    """
    payoffArrayFromDict
    Function that, given a partition in a list form and the SPFG dictionary, will return
    a list of the same length as the partition, but containing payoff instead of coalitions.
    """
    payoffs = []
    numPart = [len(i) for i in part]
    numPart.sort(reverse=True)
    while len(numPart) < n:
        numPart.append(0)
    for coal in part:
        payoffs.append(dictionary[tuple([len(coal), tuple(numPart)])])
    return payoffs

def updateDiffCheck(diff, bigDiff, failmsg, msg):
    """
    updateDiffCheck
    Facilitator function that ensures a fail is detected only if significant, and updates values.
    """
    if math.isclose(diff, 0, rel_tol=1e-5):
        diff = 0
    if diff > bigDiff:
        bigDiff = diff
        failmsg += msg
    return bigDiff, failmsg

def checkIfStableInOrigGame(optPart, dictionary, n):
    """
    checkIfStableInOrigGame
    Function that for each stability criterion F_1 to F_4, finds the utility absolute difference missing 
    for the optPart to be stable. Will return the l-infinity norm (biggest difference of all). 
    """
    bigDiff = 0
    failmsg = ""
    u2 = payoffArrayFromDict(optPart, dictionary, n)

    # F_1
    u1 = payoffArrayFromDict([[i] for i in range(n)], dictionary, n)
    updateDiffCheck(sum(u1) - sum(u2), bigDiff, failmsg, "F_1 check fails. \n")

    # F_3
    for j in range(len(optPart)):
        updateDiffCheck(sum([u1[i] for i in optPart[j]]) - u2[j], bigDiff, failmsg, "F_3 check fails. \n")
    
    # F_2
    for neighbour in pt.neighboursForF2(optPart):
        u1 = payoffArrayFromDict(neighbour, dictionary, n)
        updateDiffCheck(sum([u1[i] for i in pt.compareCoals(neighbour, optPart)]) - sum([u2[i] for i in pt.compareCoals(optPart, neighbour)]), bigDiff, failmsg, "F_2 check fails. \n")

    # F_4
    for neighbour in pt.neighboursForF4(optPart):
        u1 = payoffArrayFromDict(neighbour, dictionary, n)
        updateDiffCheck(sum([u1[i] for i in pt.compareCoals(neighbour, optPart)]) - sum([u2[i] for i in pt.compareCoals(optPart, neighbour)]), bigDiff, failmsg, "F_4 check fails. \n")
    return bigDiff, failmsg

# %%
def csvTestWriter():
    """
    csvTestWriter
    Function that opens a csv and prepares it for the test;
    then 1) Build a dictionary for this problem
    2) Computes the matrix A associated with this dict using least squares
    3) Solves the model looking for a social welfare state that is stable.

    Here and there, also computes the delta between the SPFG given and its representation,
    both for the utility function and for the solution returned at the end.
    """
    with open('bertrand_results.csv', 'w', newline='') as csvfile:
        testCombinations = itertools.product(range(3,15), range(20, 120, 80), range(0,10))
        reswriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for combination in testCombinations:
            dictionary = buildPFGDictBertrand(combination[0], combination[1], combination[2])
            matA, R = rc.dictToMatUsingLeastSquares(dictionary, combination[0])
            epsilon1 = evaluateEpsilon(dictionary, matA)
            spf.setParams(matA)
            optimal_partition = spf.SWPlusChecksLoop()
            epsilon2, checksmsg = checkIfStableInOrigGame(optimal_partition, dictionary, combination[0])
            print(checksmsg)
            reswriter.writerow([combination[0], combination[1], combination[2], R, optimal_partition, epsilon1, epsilon2])


# %%
if __name__ == "__main__":
    # For generation of a csv with test information performed with combinations of params
    csvTestWriter()
