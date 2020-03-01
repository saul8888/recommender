from learn.lear import *
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


ref_group=4
df = pd.read_csv('user.csv')
X = df.iloc[:, [0,1,2,3,4,5,6,7]].values

preference=1
dc = pd.read_csv('food.csv')
Y = dc.iloc[:, [1,2,3,4,5,6,7]].values


centroids,group= learning(df,ref_group)

cluster = [i for i in range(ref_group)]
for i in range(ref_group):
    cluster[i] = np.zeros([1, 7])


for i in range(ref_group):
    for j in range(len(X)):
        if group[j]==i:
            randomList = [Y[j]]
            cluster[i] = np.vstack((cluster[i], randomList))

for i in range(ref_group):
    cluster[i] = cluster[i][1:]

#---------------ver---------------#
#print('GROUP0')
#group0 = group0[1:]
#print(group0)
#print('GROUP1')
#group1 = group1[1:]
#print(group1)
#print('GROUP2')
#cluster[2] = cluster[2][1:]
#print(cluster[2])
#---------------------------------#
for i in range(ref_group):
    centroids[i]= food_preferer(cluster[i],preference)


favorite = np.zeros([1, 3])
for x in range(ref_group-1):
    randomList = [[i for i in range(3)]]
    favorite = np.vstack((favorite, randomList))
#favorite = favorite[1:]
#print(favorite)

for a in range(ref_group):
    for i in range(3):
        for j in range(7):
            if sorted(centroids[a],reverse=True)[i] == centroids[a][j]:
                favorite[a][i]=j
                #print(j)


for i in range(ref_group):
    print('favorite foods group ',i)
    #print(centroids[i])
    print(favorite[i])
    for j in range(3):
        print(dc.head(0).keys()[favorite[i][j]+1])


#---------graficos de error--------#
#elbow_method(df)
#graphic(df,group)
#----------------------------------#
#print(dc.head(0).keys()[i+1])
