import numpy as np
import pandas as pd
import sklearn
import seaborn
from sklearn.datasets import load_boston

boston_data = load_boston()
print(boston_data.keys())
print(boston_data.DESCR)