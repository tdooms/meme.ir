import time

import numpy as np
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
import polars as pl
from sklearn.manifold import TSNE
import seaborn as sns


def embed(model, boxes): return model.encode('. '.join(boxes)).tolist()


def store():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    start = time.time()

    memes = pl.read_ipc("data/memes.feather")
    memes = memes.with_columns([pl.col('boxes').apply(lambda x: embed(model, x)).alias('features')])

    # computed features in 5217.94 seconds - on my pc
    print(f"computed features in {time.time() - start:.2f} seconds")

    memes.write_ipc("data/features.feather")


def visualise():
    memes = pl.read_ipc("data/features.feather")

    # memes = memes.with_columns(
    #     [pl.col('features').map(lambda x: TSNE(n_components=2).fit_transform(x)).alias('tsne')]
    # )
    # print(memes)

    start = time.time()

    features = np.array(memes['features'].to_list())
    projected = TSNE(n_components=2).fit_transform(features)

    print(f"read features in {time.time() - start:.2f} seconds")

    # visualise the t-sne per meme name in a scatter plot with seaborn
    sns.scatterplot(x=projected[:, 0], y=projected[:, 1], hue=memes['name'])
    plt.show()


if __name__ == '__main__':
    # store()
    visualise()
