import time
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.manifold import TSNE


def embed(model, boxes): return model.encode('. '.join(boxes)).tolist()


def store():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')
    start = time.time()

    memes = pd.read_feather("data/memes.feather")
    memes["features"] = memes.apply(lambda x: embed(model, x))

    # computed features in 5217.94 seconds - on my pc
    print(f"computed features in {time.time() - start:.2f} seconds")

    memes.write_ipc("data/features.feather")


def sample():
    memes = pd.read_feather("data/features.feather")
    memes.sample(1000).reset_index(drop=True).to_feather("data/sample.feather")


def visualise():
    start = time.time()
    memes = pd.read_feather("data/sample.feather")
    print(f"read features in {time.time() - start:.2f} seconds")

    start = time.time()
    features = np.array(memes['features'].to_list())
    projected = TSNE(n_components=2).fit_transform(features)
    print(f"computed PCA in {time.time() - start:.2f} seconds")

    # visualise the t-sne per meme name in a scatter plot with seaborn
    # start = time.time()
    # sns.scatterplot(x=projected[:, 0], y=projected[:, 1], hue=memes['name'])
    # plt.legend([], [], frameon=False)
    # plt.show()
    # print(f"computed plot in {time.time() - start:.2f} seconds")


if __name__ == '__main__':
    # store()
    # sample()
    visualise()
