from c_means import c_means
import time

def Normalmain(df, labels):
    k = len(set(labels)) # no of clusters
    start = time.time()
    c_means(df, labels, k, type=0)
    end = time.time()
    print("The time of execution of above program is :",(end-start) * 10**3, "ms\n")