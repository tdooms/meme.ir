import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

memes = pd.read_feather("data/memes.feather")
# memes = memes[memes["views"] > 100].reset_index(drop=True)
# # memes = memes.sample(1_000).reset_index(drop=True)

memes['views'] = memes['views'] + 1
sns.histplot(memes, x='views', bins=50, log_scale=True)
plt.show()

memes['votes'] = memes['votes'] + 1
sns.histplot(memes, x='votes', bins=10, log_scale=True)
plt.show()

print(memes["views"].describe())
print(memes["votes"].describe())
