import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def learning(df,group):
    X = df.iloc[:, [1,2,3,4,5,6,7]].values

    kmeans5 = KMeans(n_clusters=group)
    #y_kmeans5 = kmeans5.fit_predict(X)
    kmeans5.fit_predict(X)

    centroids = kmeans5.cluster_centers_
    group=kmeans5.labels_
    #print(centroids)
    #print(group)
    return centroids,group


def elbow_method(df):
    X = df.iloc[:, [1,2,3,4,5,6,7]].values
    Error =[]
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(X)
        kmeans.fit(X)
        Error.append(kmeans.inertia_)

    plt.plot(range(1, 11), Error)
    plt.title('error')
    plt.xlabel('No of clusters')
    plt.ylabel('Error')
    plt.show()


def graphic(df,group):
    X = df.iloc[:, [1,2,3,4,5,6,7]].values
    kmeans = KMeans(n_clusters=group, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(X)
    #kmeans.fit_predict(X)
    plt.scatter(X[:,0], X[:,1])
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    plt.show()


def food_preferer(df,group):

    kmeans5 = KMeans(n_clusters=group)
    kmeans5.fit_predict(df)

    centroids = kmeans5.cluster_centers_
    #group=kmeans5.labels_
    #print(centroids)
    #print(group)
    return centroids