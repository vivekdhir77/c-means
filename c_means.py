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
def c_means(df, labelsTrue, k, type, m=2):
    # m = 2 # membership value
    dataSize = len(df) # no of data points
    c = random.sample(df, k) # random k centroids
    epsilon = 1 # stopping criteria 1
    # print(c)

    dist = [[0 for i in range(dataSize)] for j in range(k)] # 2d Distance matrix
    mem = [[0 for i in range(dataSize)] for j in range(k)] #initializing memebership matrix


    while True:
        dist = calcDistMatrix(dist, c, df, type) # calculating distance values
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

    # print("\n\n")
    labels = [0 for i in range(dataSize)]

    for i in range(dataSize):
        c = max(row[i] for row in mem)
        for j in range(k):
            if c == mem[j][i]:
                labels[i] = j+1
                break

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