import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

data = pd.read_csv("C://Users//Administrator//Desktop//ML//Facebook_Live_sellers.csv")

data["status_type"] = data["status_type"].astype('category')
# label encoding
data["status_type"] = data["status_type"].cat.codes

# create new dataset with the needed attributes
DataFrame = data.select_dtypes(include=['integer']).copy()

f1 = DataFrame['status_type'].values
f2 = DataFrame['num_reactions'].values
f3 = DataFrame["num_comments"].values
f4 = DataFrame["num_shares"].values
f5 = DataFrame["num_likes"].values
f6 = DataFrame["num_loves"].values
f7 = DataFrame["num_wows"].values
f8 = DataFrame["num_hahas"].values
f9 = DataFrame["num_sads"].values
f10 = DataFrame["num_angrys"].values

x = np.array(list(zip(f1, f7)))


# plt.scatter(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, c="black", s=7)


def dist(a, b):
    return np.linalg.norm(a - b)


k = 10
C_x = np.random.randint(0, np.max(x) - 20, size=k)
C_y = np.random.randint(0, np.max(x) - 20, size=k)

C = np.array(list(zip(C_x, C_y)), dtype=np.int)
print(C)

plt.scatter(f1, f10, c='#050505', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')

C_old = np.zeros(C.shape)

clusters = np.zeros(len(x))
error = dist(C, C_old)

while error.any() != 0:
    for i in range(len(x)):
        distances = dist(x[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster

    C_old = deepcopy(C)
    for i in range(k):
        points = [x[j] for j in range(len(x)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)

    error = dist(C, C_old)

colors = ['r', 'g', 'b', 'y', 'c', 'm', 'p']
fig, ax = plt.subplots()
for i in range(k):
    points = np.array([x[j] for j in range(len(x)) if clusters[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])

ax.scatter(C[:, 0], C[:, 1], marker="*", s=1000, c="#050505")
plt.show()
