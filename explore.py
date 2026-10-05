import pandas as pd

df = pd.read_csv('data/tickets.csv')
print(df.head())
print(df["category"].value_counts())