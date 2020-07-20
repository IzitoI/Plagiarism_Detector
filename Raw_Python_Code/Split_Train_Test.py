# import libraries
import pandas as pd
import numpy as np
import os
import pickle

with open("complete_df.pkl", "rb") as file:
    complete_df = pickle.load(file)

# print(complete_df)

with open("features_df.pkl", "rb") as file:
    features_df = pickle.load(file)


# Takes in dataframes and a list of selected features (column names)
# and returns (train_x, train_y), (test_x, test_y)
def train_test_data(complete_df, features_df, selected_features):
    """Gets selected training and test features from given dataframes, and
       returns tuples for training and test features and their corresponding class labels.
       :param complete_df: A dataframe with all of our processed text data, datatypes, and labels
       :param features_df: A dataframe of all computed, similarity features
       :param selected_features: An array of selected features that correspond to certain columns in `features_df`
       :return: training and test features and labels: (train_x, train_y), (test_x, test_y)"""

    # Joining the two dataframes
    joined_dataframe = pd.concat([complete_df, features_df], axis=1)

    # Extracting the features of the training samples and converting them into array of arrays
    train_x = joined_dataframe[selected_features][joined_dataframe["Datatype"] == "train"].values

    # Extracting the class of the training samples and storing it as a array
    train_y = complete_df["Class"][complete_df["Datatype"] == "train"].values

    test_x = joined_dataframe[selected_features][joined_dataframe["Datatype"] == "test"].values
    test_y = complete_df["Class"][complete_df["Datatype"] == "test"].values

    return (train_x, train_y), (test_x, test_y)


# Testing
test_selection = list(features_df)[:2]  # first couple columns as a test
# test that the correct train/test data is created
(train_x, train_y), (test_x, test_y) = train_test_data(complete_df, features_df, test_selection)

# Select your list of features, this should be column names from features_df
# ex. ['c_1', 'lcs_word']
selected_features = ['c_3', 'lcs_word']

# check that division of samples seems correct
# these should add up to 95 (100 - 5 original files)
# print('Training size: ', len(train_x))
# print('Test size: ', len(test_x))
# print()
# print('Training df sample: \n', train_x[:10])


# Saving dataframe into csv files
def make_csv(x, y, filename, data_dir):
    """Merges features and labels and converts them into one csv file with labels in the first column.
       :param x: Data features
       :param y: Data labels
       :param file_name: Name of csv file, ex. 'train.csv'
       :param data_dir: The directory where files will be saved
    """
    # make data dir, if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    pd.concat([pd.DataFrame(y), pd.DataFrame(x)], axis=1).to_csv("{}/{}".format(data_dir, filename), header=False,
                                                                 index=False)

    # nothing is returned, but a print statement indicates that the function has run
    print('Path created: ' + str(data_dir) + '/' + str(filename))


data_dir = 'plagiarism_data'

# make_csv(train_x, train_y, "train.csv", data_dir)
# make_csv(test_x, test_y, "test.csv", data_dir)
