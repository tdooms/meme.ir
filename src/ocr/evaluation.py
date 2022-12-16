import os

import distance
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns


def rate_texts(out_dir):
    """
    read all the text files and rate them based on the text that should be there
    according to the dataset

    we will rate using the levenshtein distance between the actual text and the
    text obtained by OCR
    :return: rating as an inverse sum of levenshtein distances
    """
    # read all the memes from the feather file to validate
    memes = pd.read_feather('../../data/memes.feather')
    # the distances of the ocr to the real meme texts
    dists = list()
    max_dists = list()
    # read all the text files
    # iterate over all the text files and rate them
    for f_name in tqdm(os.listdir(out_dir)):
        if not f_name.endswith('.txt'):
            continue
        with open(out_dir + f_name, 'r') as f:
            lines = f.readlines()
            texts = [line.split(',')[1][:-1] for line in lines]

        # find matching dataset meme
        img_hash = f_name.split('.')[0]
        meme = memes[memes['url'].str.contains(img_hash)]
        text_boxes = meme['boxes'].values[0]
        dist = 0
        max_dist = 0
        for real_text, ocr_text in zip(text_boxes, texts):
            max_dist += max(len(real_text), len(ocr_text))
            if not len(ocr_text):
                dist += len(real_text)
            else:
                dist += distance.levenshtein(real_text, ocr_text)
        # if left over text boxes then sum the lengths of them to get results
        if len(text_boxes) > len(texts):
            remaining_text_boxes = text_boxes[len(texts):]
            additional_dist = sum(len(remaining_text_box) for remaining_text_box in remaining_text_boxes)
            dist += additional_dist
            max_dist += additional_dist

        dists.append(dist)
        max_dists.append(max_dist)

    print('sum over sum:', sum(dists) / sum(max_dists))
    print('sum of dists and sum of max dists:', sum(dists), sum(max_dists))
    print('max dist:', max(dists))
    print('min dist:', min(dists))
    percentage_dists = [dist / max_dist for dist, max_dist in
                        zip(dists, max_dists) if max_dist != 0]
    print('average percentage:', np.mean(percentage_dists))

    # histogram of the weights
    sns.histplot(x=percentage_dists, bins=30)
    # show median line in red and the 'desired' distance in green
    plt.axvline(np.median(percentage_dists), 0, 0.17, color='red')
    plt.axvline(0.1, 0, 0.17, color='green')
    plt.show()

    # boxplot of the percentage distances
    sns.boxplot(percentage_dists)
    plt.show()
