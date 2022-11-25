# import json
#
# import pandas as pd
#
# top = pd.read_csv("ImgFlip575K_Dataset/dataset/popular_100_memes.csv")
# stat = json.load(open("ImgFlip575K_Dataset/dataset/statistics.json"))
#
# set1 = set(top["Name"].values)
# set2 = set(x[:-5].replace('-', ' ').replace('\'', '') for x in stat["memes"].keys())
#
# print(len(set1))
# print(len(set2))
#
# print(set1)
# print(set2)
#
# print(set1 - set2)
import pandas as pd

memes = pd.read_feather("data/memes.feather")
train = memes[memes["train"]].reset_index(drop=True).to_feather("data/train.feather")
test = memes[~memes["train"]].reset_index(drop=True).to_feather("data/test.feather")
