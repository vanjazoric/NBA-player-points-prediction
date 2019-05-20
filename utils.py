import numpy as np
import glob
from load_data import *
from prediction_algorithms import *
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

import pandas as pd


def calculate_rmse(y_predict, y_true):
    '''
    Root mean square error function. Return square root of difference of squares means predicted and real values of y

    :param y_predict: predicted values
    :param y_true: correct values
    :return: number representing a error
    '''
    return np.sqrt(((y_predict - y_true) ** 2).mean())


def get_supported_players():
    '''
    Function for finding all player names in data folder and collect them in list. Finding by .csv files, split file
    name and first token is name.

    :return: list of strings represent names of players supported predict points in this application
    '''

    supported_players = []

    files = glob.glob("dataset/*.csv")

    for file in files:

        file_name = file.split('\\')[1]
        player_name = file_name.split('.')[0]
        supported_players.append(player_name)

    return supported_players


def collect_attributes(dataFrame):
    '''
    Function for separation each column from data frame and collect them to array for processing

    :param dataFrame: data frame from pandas library, with all data rows and columns from .csv file
    :return: x and y arrays, representing training and test sets
    '''

    data_birthday = dataFrame['Birthday'].values
    data_road = dataFrame['Road'].values
    #data_mp = dataFrame['MP'].values
    data_fg = dataFrame['FG'].values
    data_fga = dataFrame['FGA'].values
    data_fgp = dataFrame['FGP'].values
    data_3p = dataFrame['3P'].values
    data_3pa = dataFrame['3PA'].values
    data_3pp = dataFrame['3PP'].values
    data_ft = dataFrame['FT'].values
    data_fta = dataFrame['FTA'].values
    data_ftp = dataFrame['FTP'].values
    data_orb = dataFrame['ORB'].values
    data_stl = dataFrame['STL'].values
    data_pf = dataFrame['PF'].values
    data_gm_sc = dataFrame['GmSc'].values
    data_ex = dataFrame['Ex'].values
    data_ch = dataFrame['ChristmasDay'].values

    data_pts = dataFrame['PTS'].values

    x0 = np.ones(len(data_pts))

    train_x = np.array([x0, data_birthday, data_road, data_fg, data_fga, data_fgp, data_3p, data_3pa, data_3pp,
                        data_ft, data_fta, data_ftp, data_orb, data_stl, data_pf, data_gm_sc,
                        data_ex, data_ch]).T

    train_y = np.array(data_pts)

    return train_x, train_y

def callingBatchGD(player):
    '''
    Function for loading data, split on test and train, calling batch gradient descent with data params,
    do the prediction and return rmse result

    This code appears multiple times in code so separate them in function because of redundancy

    :param player: string name of player, input from keyboard
    :return: RMSE metrics for given player
    '''

    train_data, test_data = load_data('dataset/' + player + '.csv')

    x, y = collect_attributes(train_data)

    newB, cost_history_retval = batch_gradient_descent(x, y, B, recommended_alpha, recommended_iteration_number)

    x_test, y_test = collect_attributes(test_data)
    y_pre = x_test.dot(newB)

    rmse = calculate_rmse(np.array(y_pre), y_test)

    print("\nRMSE for player "+player+" is: "+str(rmse)+"\n")

    return rmse

def callingStochasticGD(player):
    '''
    Function for loading data, split on test and train, calling stochastic gradient descent with data params,
    do the prediction and return rmse result

    :param player: string name of player, input from keyboard
    :return: RMSE metrics for given player
    '''

    train_data, test_data = load_data('dataset/'+ player+ '.csv')
    x, y = collect_attributes(train_data)
    x_test, y_test = collect_attributes(test_data)

    newB, cost_history_retval = stochastic_gradient_descent(x, y, B, recommended_alpha, recommended_iteration_number)

    y_pre = x_test.dot(newB.T[0])
    # y_pre = x_test.dot(newB)

    rmse = calculate_rmse(np.array(y_pre), y_test)

    print("\nRMSE for player "+player+" is: "+str(rmse)+"\n")

    return rmse

def callingMultipleLinearRegressionWithNp(player):
    '''
    Function for loading data, split on test and train, calling multiple linear with data params,
    do the prediction and return rmse result

    :param player: string name of player, input from keyboard
    :return: RMSE metrics for given player
    '''

    train_data, test_data = load_data('dataset/'+ player+ '.csv')

    x, y= collect_attributes(train_data)
    x_test, y_test = collect_attributes(test_data)

    y_pre = multiple_linear_regression_with_np(x, y, x_test, y_test)

    print(y_pre)

    rmse = calculate_rmse(np.array(y_pre), y_test)

    print("\n RMSE for player "+player+" is: "+str(rmse)+"\n")

    return rmse

def callingKNN(player):
    '''
    Function for loading data, split on test and train, calling KNN algorithm with data params,
    do the prediction and return rmse result

    :param player: string name of player, input from keyboard
    :return: RMSE metrics for given player
    '''

    train_data, test_data = load_data('dataset/' + player + '.csv')
    x, y = collect_attributes(train_data)
    x_test, y_test = collect_attributes(test_data)

   # By applying PCA, RMSE for all players is too high (about 9) because of small number of components.
   # x, x_test = pca(x, x_test)

    predicted = knn(x, y, x_test)
    print(predicted)
    rmse = calculate_rmse(np.array(predicted), y_test)
    print("\n[KNN] RMSE for player "+player+" is: "+str(rmse)+"\n")
    return rmse

def callingMultipleLinearRegression(player):
    train_data, validate_data, test_data = load_data('dataset/'+ player+ '.csv')
    x, y = collect_attributes(train_data)
    x_validate, y_validate = collect_attributes(validate_data)
    x_test, y_test = collect_attributes(test_data)

    minerr = 5
    mink=10

    list_of_errors=[]
    list_of_koefs= []
    lambdas = np.arange(0, 10, 0.1)
    for k in lambdas:
        koef = calculate_koef(x, y, k)
        list_of_koefs.append(koef)
        y_pre = x_test.dot(koef)
        err= calculate_rmse(y_pre, y_test)
        list_of_errors.append(err)
        # print(err)
        if err<minerr:
            minerr = err
            mink = k
            # print("MINIMAL:")
            # print(minerr)
            # print("K")
            # print(mink)

    # print("this is error 1")
    # print(minerr)

    koef= calculate_koef(x, y,mink)

    y_pre = x_test.dot(koef)

    err= calculate_rmse(y_pre, y_test)

    y_predict2= x_validate.dot(koef)
    err2 = calculate_rmse(y_predict2, y_validate)
    # print(y_validate)
    # print(y_predict2)
    # print("this is error 2, error for validate y")
    # print(err2)
    # print(list_of_koefs)

    print("\n RMSE for player " + player + " is: " + str(err) + "\n")

    return err

def callingSVR(player):
    train_data, test_data = load_data('dataset/' + player + '.csv')
    x, y = collect_attributes(train_data)
    x_test, y_test = collect_attributes(test_data)

    predicted = svr(x, y, x_test)
    print(predicted)
    rmse = calculate_rmse(np.array(predicted), y_test)
    print("\n[SVR] RMSE for player "+player+" is: "+str(rmse)+"\n")
    return rmse
