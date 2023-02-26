import pandas as pd
import math

def str_column_to_int(class_values):
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i+1
	return lookup



df = pd.read_csv("./dataset/lenses.csv", header= None)
print(df.head())

df.dropna()
df = df.drop(df[df=='?'].dropna(how='all').index)
df.drop(df.columns[[0]], axis=1, inplace=True)
df = df.values.tolist()
# print(len(df))
results = [x[-1] for x in df] # cluster number
lookup = str_column_to_int(results)
results = [lookup[x[-1]] for x in df] # cluster number in integers
# print(len(results))
df = [x[:-1] for x in df] # removing cluster number

df_new = []
for row in df: # row/sqrt(f1^2 + f2^2 + f3^2...)
    denominator = 0
    record = []
    for feature in row:
        # print(feature)
        x = 0
        try:
            x = (float(feature)*float(feature))
            denominator += x
        except:
            print(row, feature, x)
            exit(1)
    denominator = math.sqrt(denominator)
    for feature in row:
        record.append(float(feature)/denominator)
    df_new.append(record)

i = 0
df_old = []
for row in df: 
    row.append(results[i])
    df_old.append(row)
    i+=1
i = 0 
df = []
for row in df_new: 
    row.append(results[i])
    df.append(row)
    i+=1
df = pd.DataFrame(df)
df_old = pd.DataFrame(df_old)
# print(df.head(5))
print(len(lookup))
df_old.to_csv("./dataset/lenses.csv", index=False, header=None)
df.to_csv("./Quantum_Dataset/lenses_quant.csv", index=False, header=None)