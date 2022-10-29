import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time

from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import cluster, datasets


def kmeans(x, k, no_of_iterations):
    idx = np.random.choice(len(x), k, replace=False)

    # Step 1
    # Randomly choosing Centroids
    centroids = x[idx, :]

    # Step 2
    # finding the distance between centroids and all the data points
    distances = cdist(x, centroids, 'euclidean')

    # Step 3
    # Centroid with the minimum Distance
    points = np.array([np.argmin(i) for i in distances])

    # Step 4
    # Repeating the above steps for a defined number of iterations
    centers = []
    c = 0
    for _ in range(no_of_iterations):
        centroids = []
        for idx in range(k):
            # Updating Centroids by taking mean of Cluster it belongs to
            temp_cent = x[points == idx].mean(axis=0)
            centroids.append(temp_cent)
        # Updated Centroids
        centroids = np.vstack(centroids)
        distances = cdist(x, centroids, 'euclidean')
        points = np.array([np.argmin(i) for i in distances])
        visualize_clusters(x, points, centroids)
        equal = np.array_equal(centers, centroids)
        plt.pause(1)
        plt.clf()
        if equal:
            return points, centroids
        centers = centroids
        c += 1
        print(c)
    return points, centroids


def visualize_clusters(df, label, centroids):
    # Visualize the clusters
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
    plt.legend()
    for x in centroids:
        plt.plot([x[0]],
                 [x[1]],
                 'k+',
                 markersize=10,
                 )


def main():
    k = int(input("Numbers of Clusters k:"))
    iterations = int(input("max. Iterations:"))

    # mode = input("Choose Mode: \n [0] one run step by step visual \n [1] runtime of multiple runs\n")

    data = pd.read_csv("winequality-white.csv")

    # Principal component analysis
    pca = PCA(2)

    # Transform the data
    df = pca.fit_transform(data)

    # Applying our function
    label, centroids = kmeans(df, k, iterations)

    # Visualize the result
    visualize_clusters(df, label, centroids)
    plt.show()

    model = KElbowVisualizer(KMeans(), k=20)
    model.fit(df)
    model.show()


if __name__ == '__main__':
    main()
