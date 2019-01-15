import numpy as np
import glob
from load_data import *
from prediction_algorithms import *


def calculate_rmse(y_predict, y_true):
    '''
    Root mean square error function. Return square root of difference of squares means predicted and real values of y

    :param y_predict: predicted values
    :param y_true: correct values
    :return: number representing a error
    '''

    print(y_true)
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

    data = laod_data('dataset/' + player + '.csv')

    # Split data on test and train dataset. Ratio for test and train is 85% : 15%
    train_data = data[:int(data.shape[0] * 0.85)]
    test_data = data[int(data.shape[0] * 0.85):]

    x, y = collect_attributes(train_data)

    newB, cost_history_retval = batch_gradient_descent(x, y, B, recommended_alpha, recommended_iteration_number)

    x_test, y_test = collect_attributes(test_data)
    y_pre = x_test.dot(newB)

    print(np.array(y_pre))

    rmse = calculate_rmse(np.array(y_pre), y_test)

    print("\nRMSE for player "+player+" is: "+str(rmse)+"\n")

    return rmse
