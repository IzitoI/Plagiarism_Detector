# Plagiarism_Detector
This repository hosts the Plagiarism Detector Project of the Machine Learning Engineer Nanodegree Program (Udacity).

The core files are the Jupiter notebook (1, 2, 3). The folder Raw_Python_Code contains the python code used in the Jupiter notebooks.

This project aims to classify whether an answer_text, A, is plagiarized copying part of the text from a source_text, S, (Wikipedia source). The data set is divided into 70 training samples, 25 test samples, and 5 original files.

Reference paper: https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c412841_developing-a-corpus-of-plagiarised-short-answers/developing-a-corpus-of-plagiarised-short-answers.pdf

The performance of the model is measured based on its accuracy. For the data set above mentioned, an AWS SageMaker built-in LinearLearner method is used with an accuracy of 1.
