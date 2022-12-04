import random
import subprocess
import json
import os

import numpy as np
import pandas as pd

origin = "ImgFlip575K_Dataset/dataset"
destination = "data"


def clone_dataset():
    if os.path.exists(origin):
        print("The repository was already cloned.")
        return

    url = 'https://github.com/schesa/ImgFlip575K_Dataset.git'
    subprocess.check_call(['git', 'clone', url])


def remove_dataset():
    subprocess.check_call(['rm', '-rf', 'ImgFlip575K_Dataset'])


def extract_meme_data(path, label):
    data = json.load(open(f'{origin}/memes/{path}'))

    frame = {
        'url': [entry['url'] for entry in data],
        'label': [label] * len(data),
        'post': [entry['post'] for entry in data],
        'views': [entry['metadata']['views'] for entry in data],
        'votes': [entry['metadata']['img-votes'] for entry in data],
        'title': [entry['metadata']['title'] for entry in data],
        'author': [entry['metadata']['author'] for entry in data],
        'boxes': [entry['boxes'] for entry in data]
    }

    return pd.DataFrame(frame)


def process_templates(stats, write=True):
    data = [json.load(open(f"{origin}/templates/{path}")) for path in stats['path']]

    frame = {
        "title": [entry['title'] for entry in data],
        "url": [entry['template_url'] for entry in data],
        "alt_names": [entry['alternative_names'] for entry in data],
        "id": [entry['template_id'] for entry in data],
    }
    templates = pd.DataFrame(frame)
    if write: templates.to_csv(f'{destination}/templates.csv', index=False)
    return templates


def process_memes(stats, write=True):
    memes = pd.concat([extract_meme_data(path, label) for path, label in stats[['path', 'label']].values])

    memes['views'] = memes['views'].str.replace(',', '').astype('int32')
    memes['votes'] = memes['votes'].fillna('0').astype('int32')

    memes.info(memory_usage='deep')

    train_size = 500_000
    mask = [True] * train_size + [False] * (len(memes) - train_size)
    np.random.shuffle(mask)
    memes["train"] = mask

    memes.reset_index(drop=True, inplace=True)
    if write: memes.to_feather(f'{destination}/memes.feather')

    return memes


def process_statistics(write=True):
    stats = json.load(open(f'{origin}/statistics.json'))
    stats = stats["memes"]

    stats = pd.DataFrame({'path': list(stats.keys()), 'count': list(stats.values())})
    stats = stats.sort_values(by='count', ascending=False)
    stats["category"] = manual_categories()
    stats["label"] = list(range(len(stats)))

    if write: stats.to_csv(f'{destination}/statistics.csv', index=False)
    return stats


def manual_categories():
    return ['PS', 'SP', 'PS', 'SS', 'SP', 'PS', 'SP', 'PS', 'PS', 'SP', 'SS', 'SS', 'SP', 'PS', 'PS', 'PS', 'SP', 'PS',
            'SP', 'PS', 'SP', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'SP', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'SS', 'PS',
            'PS', 'SP', 'SS', 'SP', 'PS', 'SP', 'PS', 'PS', 'SS', 'SP', 'SP', 'SP', 'SS', 'PS', 'PS', 'PS', 'PS', 'PS',
            'PS', 'PS', 'PS', 'SP', 'PS', 'SP', 'PS', 'PS', 'SP', 'PS', 'SP', 'SS', 'SS', 'SP', 'PS', 'PS', 'SP', 'SP',
            'PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'SP', 'PS', 'SP', 'SP', 'PS', 'PS', 'PS', 'PS', 'PS', 'PS', 'SP', 'SP',
            'PS', 'PS', 'SP', 'SP', 'SS', 'SP', 'SP', 'PS', 'PS']


def main():
    clone_dataset()
    if not os.path.exists(destination): os.makedirs(destination)

    stats = process_statistics()
    print(process_templates(stats))
    print(process_memes(stats))

    # remove_dataset()


if __name__ == '__main__':
    pd.options.display.max_rows = 100
    pd.options.display.max_columns = None
    pd.options.display.width = None
    main()
