import numpy as np 
import pandas as pd 
import random
import sklearn.datasets
import sklearn.cluster
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import f1_score
import time

from utils import *
def main(df, labelsTrue):
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
    
    k = 3 # no of clusters
    m = 2 # membership value
    dataSize = len(df) # no of data points
    c = random.sample(df, k) # random k centroids
    epsilon = 1 # stopping criteria 1
    # print(c)

    dist = [[0 for i in range(dataSize)] for j in range(k)] # 2d Distance matrix
    mem = [[0 for i in range(dataSize)] for j in range(k)] #initializing memebership matrix


    while True:
        dist = calcDistMatrix(dist, c, df, type=0) # calculating distance values
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
    # print("\n\n")
    #print(mem)

    # print("\n\n")
    labels = [0 for i in range(dataSize)]

    for i in range(dataSize):
        c = max(row[i] for row in mem)
        if c == mem[0][i]:
            labels[i] = 1
        elif c == mem[1][i]:
            labels[i] = 2
        else:
            labels[i] = 3

    nplabelsTrue = np.array(labelsTrue)
    nplabels = np.array(labels) # predicted labels
    X = np.array(df) # dataframe in numpy (without labels)

    ari = adjusted_rand_score(nplabelsTrue, labels)
    F1_score = f1_score(nplabelsTrue, labels, average='macro')
    nmi = normalized_mutual_info_score(nplabelsTrue, labels)
    score1 = sklearn.metrics.silhouette_score(X, nplabels, metric='euclidean')
    score2 = sklearn.metrics.davies_bouldin_score(X,nplabels)
    print("ARI: ",ari)
    print("NMI: ",nmi)
    print("F1_score: ",F1_score)
    print("SI: ",score1)
    print("DBS: ",score2)

if __name__ == "__main__":
    df = pd.read_csv('./dataset/wine.csv')
    df.drop(df.columns[[0]], axis=1, inplace=True)
    df = df.values.tolist()
    labels = [x[-1] for x in df] # cluster labels
    df = [x[:-1] for x in df]
    start = time.time()
    main(df, labels)
    end = time.time()
    print("The time of execution of above program is :",(end-start) * 10**3, "ms")