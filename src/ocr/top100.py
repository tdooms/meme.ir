import os

import pandas as pd

memes = pd.read_feather("../../data/memes.feather")
memes = memes[['url', 'votes']]
memes = memes.sort_values('votes', ascending=False).reset_index(drop=True)
top_100 = memes.head(100)
meme_urls = top_100['url'].values

if not os.path.exists('../../top100'):
    os.makedirs('../../top100')

for meme_url in meme_urls:
    meme_name = meme_url.split('/')[-1]
    os.system(f'wget {meme_url} -O top100/{meme_name}')
