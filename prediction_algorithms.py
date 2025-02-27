import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR

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

def knn(x, y, x_test):
    knn = KNeighborsClassifier(n_neighbors=4, weights='distance')
    knn.fit(x,y)
    return knn.predict(x_test)


def pca(x_train, x_test):
    pca = PCA(svd_solver='full', n_components=4)
    x_train = pca.fit_transform(x_train)
    x_test = pca.fit_transform(x_test)
    return x_train, x_test

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

def multiple_linear_regression_with_np(X, Y, x_test, y_test):
    regr= linear_model.LinearRegression()
    regr.fit(X, Y)
    regr.score(X, Y)
    y_pred= regr.predict(x_test)

    # variance score: 1 means perfect prediction
    print('Variance score: {}'.format(regr.score(x_test, y_test)))

    # print('Intercept: \n', regr.intercept_)
    # print('Coefficients: \n', regr.coef_)
    return y_pred

def calculate_koef(X, Y, lamb):
#Doing by the formula A = (( X^T * X )^-1) * ( X^T * Y ), A have coefficients of regression, set of coefficients
    i = np.identity(18)
    r = len(Y)
    newY = np.zeros((r,1))

    #Matrix Y
    for k in range(0,r):
        newY[k] = Y[k]

    #Calculating first part of formula
    a = np.matrix(np.add(np.dot(X.T,X),lamb*i))
    # a = np.matrix(np.dot(X.T,X))

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
    b = np.dot(X.T,Y)
    
    #Final result of function
    result = np.dot(a,b)

    return result.tolist()[0]


def svr(x, y, x_test):
    '''
        :param C: Penalty parameter C of the error term
        :param gamma: Kernel coefficient
        :param epsilon: specifies the epsilon-tube within which no penalty is
               associated in the training loss function with points
               predicted within a distance epsilon from the actual value.
        :param kernel: the kernel type used in the algorithm
        :param degree: degree of the polynomial kernel function (‘poly’).
        '''
    clf = SVR(gamma='auto', C=0.11, epsilon=0.3, kernel='poly', degree=2)
    clf.fit(x, y)
    return clf.predict(x_test)

