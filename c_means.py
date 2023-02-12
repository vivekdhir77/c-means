import numpy as np 
import pandas as pd 
import random
import sklearn.datasets
import sklearn.cluster
# Load IRIS dataset
#

from utils import *
# df = [
#     [5.1, 3.5, 1.4, 0.2],
#     [4.9, 3, 1.4, 0.2],
#     [4.7, 3.2, 1.3, 0.2],
#     [7, 3.2, 4.7, 1.4],
#     [6.4, 3.2, 4.5, 1.5],
#     [6.9, 3.1, 4.9, 1.5],
#     [5.6, 2.7, 4.2, 1.3],
#     [6.3, 3.3, 6, 2.5],
#     [5.8, 2.7, 5.1, 1.9],
#     [7.1, 3, 5.9, 2.1],
#      ] # points

df = pd.read_csv('iris.csv')
df = df.values.tolist()
df = [x[:-1] for x in df]
k = 3 # no of clusters
m = 2 # membership value
dataSize = len(df) # no of data points
c = random.sample(df, k) # random k centroids
epsilon = 1 # stopping criteria 1
# print(c)

dist = [[0 for i in range(dataSize)] for j in range(k)] # 2d Distance matrix
mem = [[0 for i in range(dataSize)] for j in range(k)] #initializing memebership matrix


while True:
    dist = calcDistMatrix(dist, c, df) # calculating distance values
    mem = caclMembershipMatrix(mem ,dist ,dataSize ,k, m) # calculating membership values
    # print(dist)
    #mem_final = [mem[0][i]+mem[1][i]+mem[2][i] for i in range(len(mem[0]))] 
    # print(mem_final)
    new_centroids = calcNewCentroid(df, mem, dataSize, k, m)
    # print(new_centroids)
    obj_func = calcObjFunction(df,mem, new_centroids, dataSize, k, m) # objective funtion between new centroids and datapoints
    old_obj_func = calcObjFunction(df,mem, c, dataSize, k, m) # objective function between old centroids and datapoints
    norm_obj_fun = abs(old_obj_func-obj_func)
    # print(norm_obj_fun)
    center_diff = [math.sqrt(norm_square(new_centroids[i], c[i])) for i in range(k)]
    # print(center_diff)
    if stoppingCriteria2(center_diff) or norm_obj_fun<epsilon:
        break
    c = new_centroids
#print(new_centroids)
print("\n\n")
#print(mem)

print("\n\n")
labels = [0 for i in range(dataSize)]

for i in range(dataSize):
    c = max(mem[0][i],mem[1][i],mem[2][i])
    if c == mem[0][i]:
        labels[i] = 0
    elif c == mem[1][i]:
        labels[i] = 1
    else:
        labels[i] = 2

nplabels = np.array(labels)

#print(nplabels)
X = np.array(df)
score1 = sklearn.metrics.silhouette_score(X, nplabels, metric='euclidean')
score2 = sklearn.metrics.davies_bouldin_score(X,nplabels)
print(score1)
print(score2)