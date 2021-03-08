# uses multiple regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pylab as pl

dataset = pd.read_csv("C://Users//Administrator//Desktop//ML//forestfires.csv")

#print(dataset.shape)
#print(dataset.head())
#print(dataset.dtypes)

# selecting only the integer type data columns for feeding it into the learning model
final_data = dataset.select_dtypes(include=['integer', 'float']).copy()
print(final_data.shape)
print(final_data.dtypes)
