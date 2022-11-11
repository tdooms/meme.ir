import polars as pl

print(pl.read_ipc("data/memes.feather"))
print(pl.read_csv("data/templates.csv"))
print(pl.read_csv("data/statistics.csv"))
