import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.cluster import k_means_

dataset = pd.read_csv("C://Users//Administrator//Desktop//ML//AllBooks_baseline_DTM_Labelled.csv")



k = 10
samples, features = dataset.shape

print(features)
print(samples)
print(dataset.head())