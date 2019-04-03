#THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING A TUTOR OR CODE
#WRITTEN BY OTHER STUDENTS - NAOMI KEUSCH BAKER AND DORIS ZHOU

#Sources:
#http://stanford.edu/~cpiech/cs221/handouts/kmeans.html
#https://www.geeksforgeeks.org/python-pandas-dataframe/
#https://mubaris.com/posts/kmeans-clustering/

import numpy
import pandas
import random
import sys

MAX_ITERATIONS = 10

def main():
    dataSet = sys.argv[1]
    k = int(sys.argv[2])
    outPut = sys.argv[3]
    df_full = pandas.read_csv(dataSet, sep=',', header=None)
    df = df_full.drop(df_full.columns[4], axis=1)
    #print(df)
    kmeans(df, k)

# Function: K Means
# -------------
# K-Means is an algorithm that takes in a dataset and a constant
# k and returns k centroids (which define clusters of data in the
# dataset which are similar to one another).
def kmeans(dataSet, k):
    # Initialize centroids randomly
    numObjects = max(dataSet.index.values)
    centroids = getRandomCentroids(dataSet, k)
    #print(centroids)
    # Initialize book keeping vars.
    iterations = 0
    oldCentroids = None

    # Run the main k-means algorithm
    # while not shouldStop(oldCentroids, centroids, iterations):
        # # Save old centroids for convergence test. Book keeping.
        # oldCentroids = centroids
        # iterations += 1

        # # Assign labels to each datapoint based on centroids
    labels = getLabels(dataSet, centroids, k)
    #print(labels)

        # # Assign centroids based on datapoint labels
    centroids = getCentroids(dataSet, labels, numObjects, k)

    # # We can get the labels too by calling getLabels(dataSet, centroids)
    return centroids


def getRandomCentroids(dataSet, k):
    return dataSet.sample(3)


# Function: Should Stop
# -------------
# Returns True or False if k-means is done. K-means terminates either
# because it has run a maximum number of iterations OR the centroids
# stop changing.
def shouldStop(oldCentroids, centroids, iterations):
    if iterations > MAX_ITERATIONS: return True
    return oldCentroids == centroids


# Function: Get Labels
# -------------
# Returns a label for each piece of data in the dataset.
def getLabels(df, centroids, k):
    # For each element in the dataset, chose the closest centroid.
    # Make that centroid the element's label.
    numObjects = max(df.index.values)
    labels = []
    for x in range(numObjects):
        first = numpy.sqrt(((centroids - df.iloc[x])**2).sum(axis=1))
        list = first.tolist()
        result = min(first)
        labels.extend([list.index(result)])
    return labels


    # Function: Get Centroids
    # -------------
    # Returns k random centroids, each of dimension n.
def getCentroids(df, labels, numObjects, k):
    # Each centroid is the geometric mean of the points that
    # have that centroid's label. Important: If a centroid is empty (no points have
    # that centroid's label) you should randomly re-initialize it.
    
    for i in range(k):
        points = [df.iloc[j] for j in range(numObjects) if labels[j] == i]
        labels[i] = numpy.mean(points, axis = 1)
    print(labels)

if __name__ == '__main__':
    main()
