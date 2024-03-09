"""
This Python file is example of how your `pred.py` script should
look. Your file should contain a function `predict_all` that takes
in the name of a CSV file, and returns a list of predictions.

Your `pred.py` script can use different methods to process the input
data, but the format of the input it takes and the output your script produces should be the same.

Here's an example of how your script may be used in our test file:

    from example_pred import predict_all
    predict_all("example_test_set.csv")
"""

# basic python imports are permitted
import sys
import csv
import random

# numpy and pandas are also permitted
import numpy
import pandas

from model import MLPModel as Model


def predict(x):
    """
    Helper function to make prediction for a given input x.
    This code is here for demonstration purposes only.
    """
    # randomly choose between the four choices: 'Dubai', 'Rio de Janeiro', 'New York City' and 'Paris'.
    # NOTE: make sure to be *very* careful of the spelling/capitalization of the cities!!
    y = random.choice(['Dubai', 'Rio de Janeiro', 'New York City' ,'Paris'])

    a = Model()



    # return the prediction
    return y


def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    # read the file containing the test data
    # you do not need to use the "csv" package like we are using
    # (e.g. you may use numpy, pandas, etc)
    data = csv.DictReader(open(filename))

    predictions = []
    for test_example in data:
        # obtain a prediction for this test example
        pred = predict(test_example)
        predictions.append(pred)

    return predictions

p = predict_all("C:\\Users\\Aaditya\\Desktop\\University\\CSC311\\challenge\\ML_Challenge\\output.csv")
print(p)