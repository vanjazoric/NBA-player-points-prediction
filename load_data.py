import pandas as pd


def load_data(file):
    '''
    Function for reading .csv file, drop rows with NA values, encode some columns and return data for prediction

    :param file: relative path to the .csv file with data
    :return: data frame with all data, rows and columns
    '''

    df = pd.read_csv(file)
    df.dropna(inplace=True)
    #columns_to_encode = ['Road']
    #df_encoded = pd.get_dummies(data=df, columns=columns_to_encode)
    #le = preprocessing.LabelEncoder()
    #le.fit(df.dvcat)
    #le.transform(df.dvcat)

    return df