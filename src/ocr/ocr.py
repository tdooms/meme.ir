import glob
import seaborn as sns
import matplotlib.pyplot as plt
import os

import distance
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from color import get_pxls
from processing import rotate_img, \
    swt_transform
from evaluation import rate_texts
from processing import clean_img
from textbox import text_box, get_deskew_angle
from text import clean_text


def load_images(location):
    imgs = dict()
    for file_location in glob.glob(location):
        img = np.array(Image.open(file_location))
        imgs[file_location] = img

    return imgs


def show_img(img):
    im = Image.fromarray(img)
    im.show()


def ocr_images(imgs):
    char_whitelist = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:%;!?- "
    custom_config = fr"--oem 3 --psm 12 -c tessedit_char_whitelist='{char_whitelist}'"
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    img_texts = dict()
    for f_name, img in imgs.items():
        text = pytesseract.pytesseract.image_to_string(img, lang='eng',
                                                       config=custom_config)
        img_texts[f_name] = text
    return img_texts


def generate_top_100_ocr(processing_method=None, out_dir='out/'):
    # one image doesn't work
    imgs = load_images('../../top100/*.jpg')
    # iterate over all the images
    for f_name, img in tqdm(imgs.items()):
        try:
            if processing_method == 'SWT':  # apply SWT
                img = swt_transform(f_name)
            elif processing_method == 'clean':  # apply custom cleaning
                img = clean_img(img)
            sub_imgs = text_box(img)

            # rotate all the images and store them in an ocr dict
            sub_imgs_ocr = dict()
            for sub_img_idx, sub_img in sub_imgs.items():
                white_pxls = get_pxls(sub_img, [255, 255, 255])
                # do not rotate image if no white pixels are present
                if len(white_pxls):
                    deskew_angle = get_deskew_angle(white_pxls)
                    sub_img = rotate_img(sub_img, deskew_angle)

                # store in rotated textbox image in dict
                sub_imgs_ocr[sub_img_idx] = sub_img

            ocr_results = ocr_images(sub_imgs_ocr)

            # clean OCR results
            final_ocr_results = dict()
            count_idx = 0
            for textbox_idx, value in ocr_results.items():
                text = clean_text(value)
                if len(text) == 0:
                    continue
                final_ocr_results[count_idx] = text
                count_idx += 1

            new_f_name = f_name.split('/')[-1]
            write_texts(final_ocr_results, out_dir + new_f_name + '.txt')

            print(f_name)
            print('-------------------------------------------------------')
            print(final_ocr_results)
        except Exception as e:
            # if no results then write empty dict
            new_f_name = f_name.split('/')[-1]
            write_texts({}, out_dir + new_f_name + '.txt')

            print('-------------------------------------------------------')
            print(e)
            print('-------------------------------------------------------')


def fix_texts(out_dir):
    """
    fix the text files based on the typical nr of text prompts in the meme, and
    write them back to the original file
    :return: nothing
    """
    # read all the memes from the feather file to validate
    memes = pd.read_feather('../../data/memes.feather')
    # get counts of the text prompts
    memes['length'] = memes['boxes'].apply(lambda x: len(x))
    # get the most common nr of text prompts
    memes = memes.join(
        memes.groupby('title')['length'].median().rename('median_length'),
        on='title', rsuffix='')
    # read all the text files
    for f_name in tqdm(os.listdir(out_dir)):
        if not f_name.endswith('.txt'):
            continue
        with open(out_dir + f_name, 'r') as f:
            lines = f.readlines()
            texts = [line.split(',')[1][:-1] for line in lines]

        # check how many texts there are for specific meme format
        # get meme format
        img_hash = f_name.split('.')[0]
        meme_format = \
            memes[memes['url'].str.contains(img_hash)]['title'].values[0]
        # get nr of text prompts
        nr_text_prompts = \
            int(memes[meme_format == memes['title']]['median_length'].values[0])
        # remove the text prompts with the least amount of text first
        sorted_texts = sorted(texts, key=lambda x: len(x), reverse=True)
        # take only largest texts
        filtered_texts = sorted_texts[:nr_text_prompts]

        # reorder the filtered texts based on the original order
        idx_texts = sorted(
            [texts.index(filtered_text) for filtered_text in filtered_texts])
        ordered_texts = [texts[idx] for idx in idx_texts]

        # write the texts back to the file
        with open(out_dir + f_name, 'w') as f:
            for idx, text in enumerate(ordered_texts):
                f.write(str(idx) + ',' + text + '\n')


def write_texts(texts, file):
    with open(file, 'w') as f:
        for idx, text in texts.items():
            f.write(str(idx) + ',' + text + '\n')
    f.close()


if __name__ == '__main__':
    # method = 'clean'
    # generate_top_100_ocr(method, '../../out-processing/')
    # fix_texts('../../out-processing/')
    # rate_texts('../../out-processing/')
