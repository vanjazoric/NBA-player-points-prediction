from utils import *
from load_data import laod_data
from prediction_algorithms import *


def main():
    '''
    Main menu of application

    '''

    players = get_supported_players()

    print('Welcome to NBA player point predict application ')
    print('This version of application support '+str(len(players))+' different NBA players. \n\n\n')

    rmse_sum = 0

    for player in players:
        rmse_sum += callingBatchGD(player)

    print('\nGlobal deviation for all players is: \n\n'+str(rmse_sum/len(players)))

    print('\nThank you for using this application.')


if __name__ == '__main__':
    main()
