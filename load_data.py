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

    # plt.show()

    ################################################################
    # df.sort_values('PTS')
    # list1 = df[:int(df.shape[0] * 0.10)]
    # list2= df[int(df.shape[0] * 0.10):int(df.shape[0] * 0.20)]
    # list3 = df[int(df.shape[0] * 0.20):int(df.shape[0] * 0.30)]
    # list4 = df[int(df.shape[0] * 0.30):int(df.shape[0] * 0.40)]
    # list5 = df[int(df.shape[0] * 0.40):int(df.shape[0] * 0.50)]
    # list6 = df[int(df.shape[0] * 0.50):int(df.shape[0] * 0.60)]
    # list7 = df[int(df.shape[0] * 0.60):int(df.shape[0] * 0.70)]
    # list8 = df[int(df.shape[0] * 0.70):int(df.shape[0] * 0.80)]
    # list9 = df[int(df.shape[0] * 0.80):int(df.shape[0] * 0.90)]
    # list10 = df[int(df.shape[0] * 0.90):]
    #
    # list1.append(list2)
    # list1.append(list7)
    # list1.append(list10)
    # list1.append(list9)
    # list1.append(list4)
    # # list1.append(list6)
    # # list1.append(list5)
    #
    # # data on test, validate and train dataset.
    # validate_data= list3.append(list6) #df[:int(df.shape[0] * 0.20)]
    # # print(validate_data)
    # train_data = list1 #df[int(df.shape[0] * 0.20):int(df.shape[0] * 0.80)]
    # # print(train_data)
    # test_data = list8.append(list5)# df[int(df.shape[0] * 0.80):]
    # # print(test_data)
    #
    # return train_data, validate_data, test_data
    ################################################################

    return train_data, test_data