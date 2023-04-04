import csv
import pandas as pd
from dfsqroot_Quant_C_means import Quantmain
from NormalMain import Normalmain

if __name__ == "__main__":
    files = ["glymaLee.csv", "glymaWm82.csv", "glymaZh13.csv", "glysoPl.csv", "glysoW05", "pdbaa.csv", "swissprot.csv"]
    path = "./Preprocessed/"
    # files = ["lenses.csv"]
    # path = "./dataset/"
    for file in files:
        path = "./Preprocessed/"+file
        df = pd.read_csv(path)
        df.drop(df.columns[[0]], axis=1, inplace=True)
        df = df.values.tolist()
        labels = [x[0] for x in df] # cluster labels
        df = [x[1:] for x in df] # removing cluster number
        print("FILE: ", file, "\nQuantum\n")
        Quantmain(df, labels)
        print("\n\nNormal\n")
        Normalmain(df, labels)
        print("\n\n")


