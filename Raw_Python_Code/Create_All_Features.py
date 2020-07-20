# import libraries
import pandas as pd
import numpy as np
import os
import pickle

from FE_Containment import calculate_containment
from FE_LCS import lcs_norm_word


with open("complete_df.pkl", "rb") as file:
    complete_df = pickle.load(file)


# Function returns a list of containment features, calculated for a given n
# Should return a list of length 100 for all files in a complete_df
def create_containment_features(df, n, column_name=None):
    containment_values = []

    if (column_name == None):
        column_name = 'c_' + str(n)  # c_1, c_2, .. c_n

    # iterates through dataframe rows
    for i in df.index:
        file = df.loc[i, 'File']
        # Computes features using calculate_containment function
        if df.loc[i, 'Category'] > -1:
            c = calculate_containment(df, n, file)
            containment_values.append(c)
        # Sets value to -1 for original tasks
        else:
            containment_values.append(-1)

    print(str(n) + '-gram containment features created!')
    return containment_values


# Function creates lcs feature and add it to the dataframe
def create_lcs_features(df, column_name='lcs_word'):
    lcs_values = []

    # iterate through files in dataframe
    for i in df.index:
        # Computes LCS_norm words feature using function above for answer tasks
        if df.loc[i, 'Category'] > -1:
            # get texts to compare
            answer_text = df.loc[i, 'Text']
            task = df.loc[i, 'Task']
            # we know that source texts have Class = -1
            orig_rows = df[(df['Class'] == -1)]
            orig_row = orig_rows[(orig_rows['Task'] == task)]
            source_text = orig_row['Text'].values[0]

            # calculate lcs
            lcs = lcs_norm_word(answer_text, source_text)
            lcs_values.append(lcs)
        # Sets to -1 for original tasks
        else:
            lcs_values.append(-1)

    print('LCS features created!')
    return lcs_values


# Define an ngram range
ngram_range = range(1, 7)


features_list = []

# Create features in a features_df
all_features = np.zeros((len(ngram_range)+1, len(complete_df)))

# Calculate features for containment for ngrams in range
i = 0
for n in ngram_range:
    column_name = 'c_'+str(n)
    features_list.append(column_name)
    # create containment features
    all_features[i] = np.squeeze(create_containment_features(complete_df, n))
    i += 1

# Calculate features for LCS_Norm Words
features_list.append('lcs_word')
all_features[i]= np.squeeze(create_lcs_features(complete_df))

# create a features dataframe
features_df = pd.DataFrame(np.transpose(all_features), columns=features_list)

# Print all features/columns
print()
print('Features: ', features_list)
print()


with open("features_df.pkl", "wb") as file:
    pickle.dump(features_df, file)

# print some results
features_df.head(10)

# Create correlation matrix for just Features to determine different models to test
corr_matrix = features_df.corr().abs().round(2)

# display shows all of a dataframe
print(corr_matrix)
