import numpy
from numpy.linalg import norm


def topsis(matrix, weights, has_positiv_effect):
    # 1
    normalized_matrix = vector_normalization(matrix)

    # 2
    weighted_matrix = vector_normalization(weights) * normalized_matrix

    # 3
    # if positiv_effect max value for ideal_action, else min
    ideal_action = []
    anti_ideal_action = []
    for index in range(weighted_matrix.shape[1]):
        column = weighted_matrix[:, index]

        if has_positiv_effect[index] == 1:
            ideal_action.append(numpy.amax(column))
            anti_ideal_action.append(numpy.amin(column))
        elif has_positiv_effect[index] == 0:
            ideal_action.append(numpy.amin(column))
            anti_ideal_action.append(numpy.amax(column))
        else:
            raise ArithmeticError('Wrongfull input for has_positiv_effect')

    # 4
    distance_ideal_action = norm(weighted_matrix - ideal_action, axis=1)
    distance_anti_ideal_action = norm(weighted_matrix - anti_ideal_action, axis=1)

    # 5
    relative_closness = (distance_anti_ideal_action / (distance_ideal_action + distance_anti_ideal_action))

    return relative_closness


def vector_normalization(matrix):
    return matrix / norm(matrix, axis=0)
