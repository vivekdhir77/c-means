from c_means import c_means
import time
import math

def Quantmain(df, labels):
    df_new = []
    for row in df: # row/sqrt(f1^2 + f2^2 + f3^2...)
        denominator = 0
        for feature in row:
            denominator += (feature*feature)
        denominator = math.sqrt(denominator)
        record = []
        for feature in row:
            try:
                record.append(feature/denominator)
            except:
                record.append(0)
        df_new.append(record)
    df = df_new
    k = len(set(labels)) # no of clusters
    start = time.time()
    c_means(df, labels, k, type=1)
    end = time.time()
    print("The time of execution of above program is :",(end-start) * 10**3, "ms")