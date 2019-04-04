# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING A TUTOR OR CODE
# WRITTEN BY OTHER STUDENTS - NAOMI KEUSCH BAKER AND DORIS ZHOU

# Sources:
# http://stanford.edu/~cpiech/cs221/handouts/kmeans.html
# https://www.geeksforgeeks.org/python-pandas-dataframe/
# https://mubaris.com/posts/kmeans-clustering/

import numpy
import pandas
import sys
from sklearn import metrics

MAX_ITERATIONS = 10


def main():
    # read input arguments
    dataSet = sys.argv[1]
    k = int(sys.argv[2])
    output = sys.argv[3]

    # read and process dataset
    dataset_full = pandas.read_csv(dataSet, sep=',', header=None)
    dataset = dataset_full.drop(len(dataset_full.columns)-1, axis=1)  # remove the class label column from the data
    dataset = dataset.to_numpy()

    # run kmeans algorithm
    kmeans(dataset, k, output)


# Function: K Means
# -------------
# K-Means is an algorithm that takes in a dataset and a constant
# k and returns k centroids (which define clusters of data in the
# dataset which are similar to one another).
def kmeans(dataset, k, output):
    # Initialize centroids randomly
    centroids = get_random_centroids(dataset, k)

    # Initialize book keeping vars.
    iterations = 0
    old_centroids = None
    labels = []

    print("ITERATION " + str(iterations))
    print("Centroids: " + str(centroids) + "\n")

    # Run the main k-means algorithm
    while not should_stop(old_centroids, centroids, iterations):
        # Save old centroids for convergence test. Book keeping.
        old_centroids = centroids
        iterations += 1

        # Assign labels to each datapoint based on centroids
        labels = get_labels(dataset, centroids, k)

        # Re-assign centroids to be the average of all points in a cluster
        centroids = get_centroids(dataset, labels, k)

        print("ITERATION " + str(iterations))
        print("Centroids: " + str(centroids) + "\n")

    # write final clustering results to output
    with open(output, 'w') as output_file:
        for i in range(len(labels)):
            line_to_write = str(labels[i])  # label is the first item in each row
            for element in dataset[i]:
                line_to_write = line_to_write + " " + str(element)

            output_file.write(line_to_write + "\n")

        # calculate and write evaluation metrics
        output_file.write("SSE: " + str(get_sse(dataset, centroids, labels)) + "\n")
        output_file.write("Silhouette Coefficient: " + str(metrics.silhouette_score(dataset, labels)))

    return centroids


# Retrieves k random rows from the dataset
def get_random_centroids(dataset, k):
    random_indices = numpy.random.choice(dataset.shape[0], k, replace=False)
    return dataset[random_indices]


# Function: Should Stop
# -------------
# Returns True or False if k-means is done. K-means terminates either
# because it has run a maximum number of iterations OR the centroids
# stop changing.
def should_stop(old_centroids, centroids, iterations):
    if iterations > MAX_ITERATIONS:
        return True
    elif old_centroids is None:
        return False
    else:
        return numpy.array_equal(old_centroids, centroids)


# Function: Get Labels
# -------------
# Returns a label for each piece of data in the dataset.
def get_labels(dataset, centroids, k):
    # For each element in the dataset, chose the closest centroid.
    # Make that centroid the element's label.
    labeled_dataset = []
    for row in dataset:
        # find the closest centroid
        label = ''
        min_distance = sys.maxsize
        for i in range(len(centroids)):
            distance = numpy.linalg.norm(centroids[i] - row)
            if distance < min_distance:
                min_distance = distance
                label = i

        # assign the appropriate label to the point
        labeled_dataset.append(label)

    return labeled_dataset


# Function: Get Centroids
# -------------
# Returns k random centroids, each of dimension n.
def get_centroids(dataset, labels, k):
    # create a dict to hold all the points with the same label
    centroid_dict = {}
    for i in range(k):
        centroid_dict[i] = []
    for i in range(len(labels)):
        centroid_dict[labels[i]].append(dataset[i])

    # using the mean of all the points for each cluster, get the new centroids
    centroids = []
    for k, v in centroid_dict.items():
        centroids.append(numpy.mean(v, axis=0))

    # return the new centroids as a numpy array
    return numpy.asarray(centroids)


# Gets the sum squared error for a labeled dataset.
def get_sse(dataset, centroids, labels):
    sse = 0  # sum squared error
    for i in range(len(labels)):
        row = dataset[i]
        label = labels[i]
        centroid = centroids[label]

        # get the squared distance (difference) between the point and its centroid
        sq_diff = numpy.linalg.norm(row - centroid) ** 2
        sse += sq_diff  # add to sse

    return sse


if __name__ == '__main__':
    main()