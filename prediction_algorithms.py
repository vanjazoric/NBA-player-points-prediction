import numpy as np
from sklearn import linear_model


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

def stochastic_gradient_descent(X, Y, B, alpha, iterations):
    '''
    :param X: 17 columns with needed attributes for prediction
    :param Y: column representing the points scored in the match
    :param B: all initial coefficients are set to zero
    :param alpha: learning rate, initially set to 0.00001
    :param iterations: number of iterations
    :return: coefficients given by algorithm and cost history for every iteration
    '''
    cost_history = [0] * iterations
    m = len(Y)

    for iteration in range(iterations):
        cost= 0
        for i in range(0,m):
            X_i= X[i].reshape(1, X.shape[1])
            Y_i= Y[i].reshape(1,1)

            # Hypothesis Values
            # h = X[i].dot(B)
            h = X_i.dot(B)
            # Difference b/w Hypothesis and Actual Y
            # loss = h - Y[i]
            loss = h - Y_i
            # Gradient Calculation
            # gradient = X[i].T.dot(loss) / m
            gradient = X_i.T.dot(loss) / m
            # Changing Values of B using Gradient
            B = B - alpha * gradient
            # New Cost Value
            cost += cost_function(X_i, Y_i, B)
        cost_history[iteration] = cost
    return B, cost_history

def multiple_linear_regression_with_np(X, Y, B, alpha, iterations):
    # params_lm = {"alpha":0.1, "iterations":1000}
    regr= linear_model.LinearRegression()
    regr.fit(X, Y)
    regr.score(X, Y)

    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)
    B= regr.coef_
    return B

def calculate_koef(X, Y, lamb):
#Doing by the formula A = (( X^T * X )^-1) * ( X^T * Y ), A have coefficients of regression, set of coefficients
    i = np.identity(19)
    r = len(Y)
    newY = np.zeros((r,1))
    x = np.ones((r,1))
    #Matrix X
    xNew= np.hstack((x, X))
    
    #Matrix Y
    for k in range(0,r):
        newY[k] = Y[k]
    #Calculating first part of formula
    a = np.matrix(np.add(np.dot(xNew.T,xNew),lamb*i))
    try:
        a= a.I
    except:
        print("This is singular matrix, one for which an inverse does not exist")
        '''
        example
        [[   1,    8,   50],
        [   8,   64,  400],
        [  50,  400, 2500]]
        
        determinant is zero
        '''
    #Second part of formula
    b = np.dot(xNew.T,Y)
    # if(lamb==0):
    #     print(lamb)
    #     print(b)
    #Final result of function
    result = np.dot(a,b)

    return result.tolist()[0]