import pandas as pd
import csv
import math
df = pd.read_csv('lenses.csv')
df.drop(df.columns[[0]], axis=1, inplace=True)
df = df.values.tolist()
results = [x[-1] for x in df] # removing cluster number
df = [x[:-1] for x in df] # removing cluster number
df_new = []
for row in df: # row/sqrt(f1^2 + f2^2 + f3^2...)
    denominator = 0
    record = []
    for feature in row:
        denominator += (feature*feature)
    denominator = math.sqrt(denominator)
    for feature in row:
        record.append(feature/denominator)
    df_new.append(record)

i = 0
df = []
for row in df_new: # row/sqrt(f1^2 + f2^2 + f3^2...)
    row.append(results[i])
    df.append(row)
    i+=1
df = pd.DataFrame(df)
print(df.head(5))
df.to_csv("lenses_new.csv", index=False)