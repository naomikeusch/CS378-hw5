#THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING A TUTOR OR CODE
#WRITTEN BY OTHER STUDENTS - NAOMI KEUSCH BAKER AND DORIS ZHOU

#Sources:
#http://stanford.edu/~cpiech/cs221/handouts/kmeans.html
#https://www.geeksforgeeks.org/python-pandas-dataframe/

import numpy
import pandas

MAX_ITERATIONS = 10
df_full = pandas.read_csv('iris.data', sep=',', header=None)
df = df_full.drop(df_full.columns[4], axis=1)
numObjects = max(df.index.values)+1
def main():
    
    print(getLabels('iris.data', 0))
    
	#first = numpy.sqrt(((df - df.iloc[2])**2).sum(axis=1))
    #list = first.tolist()
    #result = min(first)
    #print(list)
    #print(result)
    #print(list.index(result))
    

# Function: K Means
# -------------
# K-Means is an algorithm that takes in a dataset and a constant
# k and returns k centroids (which define clusters of data in the
# dataset which are similar to one another).
def kmeans(dataSet, k):
    # Initialize centroids randomly
    numFeatures = dataSet.getNumFeatures()
    centroids = getRandomCentroids(numFeatures, k)

    # Initialize book keeping vars.
    iterations = 0
    oldCentroids = None

    # Run the main k-means algorithm
    while not shouldStop(oldCentroids, centroids, iterations):
        # Save old centroids for convergence test. Book keeping.
        oldCentroids = centroids
        iterations += 1

        # Assign labels to each datapoint based on centroids
        labels = getLabels(dataSet, centroids)

        # Assign centroids based on datapoint labels
        centroids = getCentroids(dataSet, labels, k)

    # We can get the labels too by calling getLabels(dataSet, centroids)
    return centroids
	
	#print(np.sqrt(((


def getRandomCentroids(numFeatures, k):
    print('')


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
def getLabels(dataSet, centroids):
    # For each element in the dataset, chose the closest centroid.
    # Make that centroid the element's label.

    # Function: Get Centroids
    # -------------
    # Returns k random centroids, each of dimension n.
    for x in range(k):
	    for y in range(numObjects):
            first = numpy.sqrt(((df.iloc[centroids(k)] - df.iloc[numObjects])**2).sum(axis=1))
            list = first.tolist()
            result = min(first)
            print(list)
            print(result)
            print(list.index(result))



def getCentroids(dataSet, labels, k):
    # Each centroid is the geometric mean of the points that
    # have that centroid's label. Important: If a centroid is empty (no points have
    # that centroid's label) you should randomly re-initialize it.
    print('')

if __name__ == '__main__':
    main()
