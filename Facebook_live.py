import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

dataset = pd.read_csv("C://Users//Administrator//Desktop//ML//Facebook_Live_sellers.csv")

dataset["status_type"] = dataset["status_type"].astype('category')
# label encoding
dataset["status_type"] = dataset["status_type"].cat.codes

# create new dataset with the needed attributes
data = dataset.select_dtypes(include=['integer']).copy()


# print(data.head())
# print(data.dtypes)


def euclidian(a, b):
    return np.linalg.norm(a - b)


def kmeans(k, epsilon=0, distance='euclidian'):
    global dist_method, index_prototype
    history_centroids = []
    if distance == 'euclidian':
        dist_method = euclidian

    data = dataset.select_dtypes(include=['integer']).copy()
    num_instances, num_features = data.shape

    prototypes = data[np.random.randint(0, num_instances - 1, size=k)]
    history_centroids.append(prototypes)

    prototypes_old = np.zeros(prototypes.shape)
    belongs_to = np.zeros(num_instances, 1)
    norm = dist_method(prototypes, prototypes_old)
    iteration = 0
    while norm > epsilon:
        iteration += 1
        norm = dist_method(prototypes, prototypes_old)

        for index_instance, instance in enumerate(data):
            dist_vec = np.zeros((k, 1))
            for index_prototype, prototype in enumerate(prototypes):
                dist_vec[index_prototype] = dist_method(prototype, instance)

            belongs_to[index_prototype, 0] = np.argmin(dist_vec)

        tmp_prototypes = np.zeros((k, num_features))

        for index in range(len(prototypes)):
            instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
            prototype = np.mean(data[instances_close], axis=0)
            tmp_prototypes[index, :] = prototype

        prototypes = tmp_prototypes

        history_centroids.append(tmp_prototypes)

    return prototypes, history_centroids, belongs_to


def plot(data, history_centroids, belongs_to):
    colors = ['red', 'green', 'blue', 'orange', 'yellow', 'pink', 'brown', 'cyan', 'black', 'purple']

    fig, ax = plt.subplots()

    for index in range(data.shape[0]):
        instance_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]

        for instance_index in instance_close:
            for i in range(10):
                ax.plot(data[instance_index][i], data[instance_index][i], (colors[index]))

        history_points = []

        for index, centroids in enumerate(history_centroids):
            for inner, item in enumerate(centroids):
                if index == 0:
                    for i in range(10):
                        history_points.append(ax.plot(item[i], 'bo')[0])

                else:
                    for i in range(10):
                        history_points[inner].set_data(item[i])
                        print("centroids {} {}".format(index, item))
                        plt.show()


def execute():
    data = dataset.select_dtypes(include=['integer']).copy()

    centroids, history_centroids, belongs_to = kmeans(10)

    plot(data, history_centroids, belongs_to)


execute()
