import numpy as np


recommended_alpha = 0.00001
B = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
recommended_iteration_number = 1000


def batch_gradient_descent(X, Y, B, alpha, iterations):
    '''

    :param X: 17 columns with needed attributes for prediction
    :param Y: column representing the points scored in the match
    :param B: all initial coefficients are set to zero
    :param alpha: learning rate, initially set to 0.00001
    :param iterations: number of iterations
    :return: coefficients given by algorithm and cost history for every iteration
    '''

    #print(X)
    cost_history = [0] * iterations
    m = len(Y)

    for iteration in range(iterations):
        # Hypothesis Values
        h = X.dot(B)
        # Difference b/w Hypothesis and Actual Y
        loss = h - Y
        # Gradient Calculation
        gradient = X.T.dot(loss) / m
        # Changing Values of B using Gradient
        B = B - alpha * gradient
        # New Cost Value
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost

    return B, cost_history


def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J