# import libraries
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer


with open("complete_df.pkl", "rb") as file:
    complete_df = pickle.load(file)


# Compute the normalized LCS given an answer text and a source text
def lcs_norm_word(answer_text, source_text):
    """Computes the longest common subsequence of words in two texts; returns a normalized value.
       :param answer_text: The pre-processed text for an answer text
       :param source_text: The pre-processed text for an answer's associated source text
       :return: A normalized LCS value
    """

    # Converting strings into lists of words
    answer_list = answer_text.split()
    source_list = source_text.split()

    a_len = len(answer_list)
    s_len = len(source_list)

    # Array (matrix) to store the computations
    matrix = np.zeros(shape=(s_len + 1, a_len + 1))

    # Building up the matrix
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == 0 or j == 0:
                matrix[i, j] = 0
            elif answer_list[j - 1] == source_list[i - 1]:
                matrix[i, j] = matrix[i - 1, j - 1] + 1  # Top-left diagonal element
            else:
                matrix[i, j] = max(matrix[i - 1, j], matrix[i, j - 1])  # Left and top element max

    # Fetching the bottom-right element
    # Normalizing the lcs value
    norm_lcs_value = matrix[-1, -1] / a_len

    return norm_lcs_value


# Run the test scenario from above
# does your function return the expected value?
A = "i think pagerank is a link analysis algorithm used by google that uses a system of weights attached to each " \
    "element of a hyperlinked set of documents "

S = "pagerank is a link analysis algorithm used by the google internet search engine that assigns a numerical " \
    "weighting to each element of a hyperlinked set of documents "

# calculate LCS
lcs = lcs_norm_word(A, S)
print('LCS = ', lcs)

# expected value test
assert lcs == 20 / 27., "Incorrect LCS value, expected about 0.7408, got " + str(lcs)

print('Test passed!')

# test on your own
test_indices = range(5)  # look at first few files

category_vals = []
lcs_norm_vals = []
# iterate through first few docs and calculate LCS
for i in test_indices:
    category_vals.append(complete_df.loc[i, 'Category'])
    # get texts to compare
    answer_text = complete_df.loc[i, 'Text']
    task = complete_df.loc[i, 'Task']
    # we know that source texts have Class = -1
    orig_rows = complete_df[(complete_df['Class'] == -1)]
    orig_row = orig_rows[(orig_rows['Task'] == task)]
    source_text = orig_row['Text'].values[0]

    # calculate lcs
    lcs_val = lcs_norm_word(answer_text, source_text)
    lcs_norm_vals.append(lcs_val)

# print out result, does it make sense?
print('Original category values: \n', category_vals)
print()
print('Normalized LCS values: \n', lcs_norm_vals)

