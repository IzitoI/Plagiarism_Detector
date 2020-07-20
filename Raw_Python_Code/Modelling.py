# import libraries
import pandas as pd
import numpy as np
import os
import pickle

train_X = pd.read_csv("plagiarism_data/train.csv", header=None).values.astype("float32")[:, 1:]
train_y = pd.read_csv("plagiarism_data/train.csv", header=None).values.astype("float32")[:, 0]

print(train_y)

