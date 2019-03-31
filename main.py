from utils import *
from load_data import laod_data
from prediction_algorithms import *


def main():
    '''
    Main menu of application

    '''

    players = get_supported_players()

    print('Welcome to NBA player point predict application ')
    print('This version of application support ' + str(len(players)) + ' different NBA players. \n\n\n')

    print('Please, select mode of our application: \n')
    print('1. Do the prediction for all players supported and return global results \n')
    print('2. Predict points for one specified player\n')

    choice = input("\n\n\nEnter menu option(1 or 2): ")

    if choice == '1':

        rmse_sum = 0

        for player in players:
            # rmse_sum += callingBatchGD(player)
            rmse_sum += callingStochasticGD(player)
            # rmse_sum += callingMultipleLinearRegression(player)
            # rmse_sum += callingMultipleLinearRegressionWithNp(player)

        print('\nGlobal deviation for all players is: \n\n' + str(rmse_sum / len(players)))

    elif choice == '2':

        not_found = True

        while not_found:
            print('Supported players for prediction:\n')

            for player in players:
                print('>> ' + str(player))

            player = input("\n\n\nInput player's name for prediction: ")

            if player not in players:
                not_found = True
                print('\nThis player is not supported in this version.')
            else:
                not_found = False
                print('\n\nTotal points of ' + player + ' are successfully predicted, with deviation of: '
                    #   + str(callingBatchGD(player)) + ' points.')
                    #   + str(callingStochasticGD(player)) + ' points.')
                    #   + str(callingMultipleLinearRegression(player)) + ' points.')
                      + str(callingMultipleLinearRegressionWithNp(player)) + ' points.')

    else:
        print("\nWrong menu option, start application again and enter valid option...\n")

    print('\nThank you for using this application.')


if __name__ == '__main__':
    main()
