# import libraries
import pandas as pd
import numpy as np
import os
import helpers
import pickle


# csv_file = 'data/file_information.csv'
# plagiarism_df = pd.read_csv(csv_file)

# print out the first few rows of data info
# print(plagiarism_df.head())


# Read in a csv file and return a transformed dataframe
def numerical_dataframe(csv_file='data/file_information.csv'):
    """Reads in a csv file which is assumed to have `File`, `Category` and `Task` columns.
       This function does two things:
       1) converts `Category` column values to numerical values
       2) Adds a new, numerical `Class` label column.
       The `Class` column will label plagiarized answers as 1 and non-plagiarized as 0.
       Source texts have a special label, -1.
       :param csv_file: The directory for the file_information.csv file
       :return: A dataframe with numerical categories and a new `Class` label column"""

    df = pd.read_csv(csv_file)

    # Converting "Category" column values to numerical values
    df["Category"].replace(["non", "heavy", "light", "cut", "orig"], [0, 1, 2, 3, -1], inplace=True)

    # Adding the "Class" column to the dataframe
    df["Class"] = [0 if type == 0 else -1 if type == -1 else 1 for type in df["Category"]]

    return df


# Informal testing, print out the results of a called function create new `transformed_df`
transformed_df = numerical_dataframe(csv_file='data/file_information.csv')

# Check that all categories of plagiarism have a class label = 1
print(transformed_df.head(10))
print(transformed_df.tail())


# create a text column
text_df = helpers.create_text_column(transformed_df)
text_df.head()

random_seed = 1 # can change; set for reproducibility

# create new df with Datatype (train, test, orig) column
# pass in `text_df` from above to create a complete dataframe, with all the information you need
complete_df = helpers.train_test_dataframe(text_df, random_seed=random_seed)

# check results
print(complete_df.head(10))

with open("complete_df.pkl", "wb") as file:
    pickle.dump(complete_df, file)
