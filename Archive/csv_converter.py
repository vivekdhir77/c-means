import pandas as pd
df = pd.read_csv('dataset/dim032.txt', sep='\s+', header=None)
df.to_csv('dataset/dim032.csv', header=None, index=False)