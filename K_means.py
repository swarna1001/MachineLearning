import pandas as pd
import numpy as np

data = pd.read_csv("C://Users//Administrator//Desktop//ML//Facebook_Live_sellers.csv")

data["status_type"] = data["status_type"].astype('category')
# label encoding
data["status_type"] = data["status_type"].cat.codes

# create new dataset with the needed attributes
dataset = data.select_dtypes(include=['integer']).copy()


class k_means:
    def __init__(self, k=10, tolerance=0.0001, max_iterations=500):
        self.centroids = {}
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        instances, features = dataset.shape

        # print(dataset.dtypes)

        for i in range(self.k):
            self.centroids[i] = dataset[i]

        for i in range(self.max_iterations):
            self.classes = {}
            for i in range(self.k):
                self.classes[i] = []
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)
                previous = dict(self.centroids)
                for classification in self.classes:
                    self.centroids[classification] = np.average(self.classes[classification], axis=0)

                    isOptimal = True
                    for centroid in self.centroids:
                        original_centroid = previous[centroid]
                        current = self.centroids[centroid]

                        if np.sum((current - original_centroid) / original_centroid * 100.0) > self.tolerance:
                            isOptimal = False

                        if isOptimal:
                            break

if __name__ == '__main__':
    km = k_means(10)
# km.fit(dataset)
