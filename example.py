import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn import metrics, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
print(dataset.keys())
#print(dataset)
print(dataset.items())
