import numpy as np
from skcriteria import DecisionMatrix
from skcriteria.agg import MABAC 

# example from https://www.researchgate.net/publication/365228811_Comparison_of_Multi-Criteria_Decision_Making_Methods_Using_The_Same_Data_Standardization_Method#fullTextFileContent

matrix = np.array([
    # Ra  MRR  
    [0.97, 25.465], # alternative 1
    [1.085, 305.577], # alternative 2
    [2.032, 1145.916], # alternative 3
    [0.746, 318.31], # alternative 4
    [0.609, 190.986], # alternative 5
    [1.001, 286.479], # alternative 6
    [0.858, 343.775], # alternative 7
    [0.326, 381.972], # alternative 8
    [1.083, 229.183] # alternative 9
])

# objetives
objectives = np.array([-1, 1])  # Minimize Ra and maximize MRR

# weights
weights = np.array([0.5, 0.5])  # Ra and MRR equally important

# decision matrix
dm = DecisionMatrix.from_mcda_data(
    matrix,
    objectives,
    weights=weights,
    alternatives=[f"Alternative {i}" for i in range(1, 10)],
    criteria=["Ra", "MRR"]
)

# using MABAC method
result = MABAC().evaluate(dm)

print("\nRanking (1 is best):")
print(result.rank_)

print("\nScores:")
print(result.extra_["score"])

print("\nBAA:")
print(result.extra_["border_approximation_area"])