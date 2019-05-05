import pandas as pd
import matplotlib.pyplot as plt


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

    # Vizualizing

    # create a figure and axis
    fig, ax = plt.subplots()

    # scatter the sepal_length against the sepal_width
    # ax.scatter(df['PTS'], df['PF'])

    # Split data on test and train dataset.
    train_data = df[:int(df.shape[0] * 0.85)]
    test_data = df[int(df.shape[0] * 0.85):]

    # '''
    # for hista
    # count the occurrence of each class
    data = df['PTS'].value_counts()
    # get x and y data
    points = data.index
    frequency = data.values
    # create bar chart
    ax.bar(points, frequency)
    #########
    # '''

    file_name = file.split('/')[1]
    title = file_name.split('.')[0]

    # set a title and labels
    ax.set_title(title)
    ax.set_xlabel('PTS')
    ax.set_ylabel('Frequency')

    #plt.show()

    return train_data, test_data