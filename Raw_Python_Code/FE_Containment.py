# import libraries
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer

with open("complete_df.pkl", "rb") as file:
    complete_df = pickle.load(file)


# Calculate the ngram containment for one answer file/source file pair in a df
def calculate_containment(df, n, answer_filename):
    """Calculates the containment between a given answer text and its associated source text.
       This function creates a count of ngrams (of a size, n) for each text file in our data.
       Then calculates the containment by finding the ngram count for a given answer text,
       and its associated source text, and calculating the normalized intersection of those counts.
       :param df: A dataframe with columns,
           'File', 'Task', 'Category', 'Class', 'Text', and 'Datatype'
       :param n: An integer that defines the ngram size
       :param answer_filename: A filename for an answer text in the df, ex. 'g0pB_taskd.txt'
       :return: A single containment value that represents the similarity
           between an answer text and its source text.
    """

    # Retrieving the row of the answer_filename from the dataframe
    file_row = df.loc[df["File"] == answer_filename, :]

    # Storing text and task of the answer_filename as strings
    # The string is contained in the first position of the pandas series (Hence .values[0])
    answer_text, task = file_row["Text"].values[0], file_row["Task"].values[0]

    # Retrieving the source text for the answer_filename task as a string from the pandas series
    source_text = df.loc[df["Datatype"] == "orig", :].loc[df["Task"] == task, "Text"].values[0]

    answer_and_source_list = [answer_text, source_text]  # We need a list of strings to pass to CountVectorizer

    vectorizer = CountVectorizer(analyzer="word", ngram_range=(n, n))
    vectorized_strings = vectorizer.fit_transform(answer_and_source_list).toarray()

    # Counting how many ngrams are in common between answer and source
    count_common_ngram = 0

    for i in range(len(vectorized_strings[0])):
        count_common_ngram += min(vectorized_strings[0][i], vectorized_strings[1][i])

    # Returning a normalized containment. The first array in vectorized_strings refers to the answer
    return count_common_ngram / np.sum(vectorized_strings[0])


# select a value for n
n = 1

# indices for first few files
test_indices = range(5)

# iterate through files and calculate containment
category_vals = []
containment_vals = []

for i in test_indices:
    # get level of plagiarism for a given file index
    category_vals.append(complete_df.loc[i, 'Category'])
    # calculate containment for given file and n
    filename = complete_df.loc[i, 'File']
    c = calculate_containment(complete_df, n, filename)
    containment_vals.append(c)


# print out result, does it make sense?
print('Original category values: \n', category_vals)
print()
print(str(n)+'-gram containment values: \n', containment_vals)
