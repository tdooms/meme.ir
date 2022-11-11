import polars as pl
# import pandas as pd

df = pl.read_csv("dataset/top100.csv")
print(df)
