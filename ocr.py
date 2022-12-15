import glob
import seaborn as sns
import matplotlib.pyplot as plt
import os

import cv2
import distance
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from blob import remove_big_blobs, get_blobs, remove_long_blobs, \
    remove_blob_in_blob, remove_empty_blobs
from color import get_pxls
from processing import bilateral_img, from_to_color, remove_color, rotate_img
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


def clean_images(imgs):
    clean_imgs = dict()
    for f_name, img in imgs.items():
        clean_imgs[f_name] = clean_img(img)
    return clean_imgs


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


def clean_img(img):
    # reduce size if width is above 800px
    if img.shape[0] > 800:
        img = cv2.pyrDown(img)

    # smooth image
    img = bilateral_img(img)

    # transform pixel to fuchsia if it is coloured
    img = remove_color(img)
    # show_img(img)
    img = from_to_color(img, [255, 0, 255], [0, 0, 0])
    # show_img(img)

    # remove blobs that are mostly empty
    white_blobs = get_blobs(img, [255, 255, 255])
    black_blobs = get_blobs(img, [0, 0, 0])
    img = remove_empty_blobs(img, white_blobs)
    img = remove_empty_blobs(img, black_blobs)
    # show_img(img)

    # removes big nonsensical blobs
    white_blobs = get_blobs(img, [255, 255, 255])
    black_blobs = get_blobs(img, [0, 0, 0])
    img = remove_big_blobs(img, black_blobs)
    # show_img(img)
    img = remove_big_blobs(img, white_blobs)
    # show_img(img)

    # remove long blobs
    white_blobs = get_blobs(img, [255, 255, 255])
    black_blobs = get_blobs(img, [0, 0, 0])
    img = remove_long_blobs(img, black_blobs)
    # show_img(img)
    img = remove_long_blobs(img, white_blobs)
    # show_img(img)

    # remove the blobs within blobs
    white_blobs = get_blobs(img, [255, 255, 255])
    black_blobs = get_blobs(img, [0, 0, 0])
    img = remove_blob_in_blob(img, black_blobs + white_blobs)
    # show_img(img)

    img = from_to_color(img, [0, 0, 0], [255, 255, 255])
    img = from_to_color(img, [255, 0, 255], [0, 0, 0])
    # show_img(img)
    return img


def text_box(img):
    # accessed on 20
    # Convert the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    # Specify structure shape and kernel size.
    # Kernel size increases or decreases the area
    # of the rectangle to be detected.
    # A smaller value like (10, 10) will detect
    # each word instead of a sentence.
    dim = img.shape[0] // 40
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                            (int(dim * 2.2), int(dim * 1.5)))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)

    # Creating a copy of image
    im2 = img.copy()

    sub_imgs = list()
    sub_img_xy = list()
    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    # to pytesseract for extracting text from it
    # Extracted text is then written into the text file
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        sub_img_xy.append((y, x))

        # Cropping the text block for giving input to OCR
        cropped = im2[y:y + h, x:x + w]
        sub_imgs.append(cropped)

    # we need to group items by approx same height so that they then can be
    # sorted from left to right
    groups = [[]]
    cur_idx = 0
    prev_xy = None
    for one in sub_img_xy:
        # if there was no previous sub image then continue after adding to group
        if prev_xy is None:
            pass
        # if current item is approx on same height as other, then put in same
        # group
        elif one[0] - 30 < prev_xy[0] < one[1] + 30:
            pass
        else:  # create new group
            groups.append([])
            cur_idx += 1

        groups[cur_idx].append(one)
        prev_xy = one

    cur_nr_group_members = 0
    reindex = list()
    for group_idx, group in enumerate(groups):
        groupx = np.array(list(map(lambda v: v[1], group)))
        groupx_idx = np.argsort(groupx)
        groupx_idx += cur_nr_group_members
        cur_nr_group_members += len(group)
        reindex.extend(list(reversed(list(groupx_idx))))
    reindex = list(reversed(reindex))
    sub_imgs = np.array(sub_imgs)
    sub_imgs = sub_imgs[reindex]

    sub_imgs_dict = dict()
    for sub_img_idx, sub_img in enumerate(sub_imgs):
        sub_imgs_dict[sub_img_idx] = sub_img
    return sub_imgs_dict


def get_deskew_angle(pxls):
    y = [xy[0] for xy in pxls]
    x = [xy[1] for xy in pxls]

    model = LinearRegression()
    model.fit(np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1))

    coef = model.coef_[0][0]

    hyp = np.sqrt(coef ** 2 + 1)

    angle = np.arcsin(coef / hyp)
    angle = np.degrees(angle)

    return angle


def show_top_100_ocr():
    # one image doesn't work
    imgs = load_images('top100/*.jpg')
    # iterate over all the images
    for f_name, img in tqdm(imgs.items()):
        try:
            img = clean_img(img)
            # show_img(img)

            sub_imgs = text_box(img)

            # rotate all the images and store them in an ocr dict
            sub_imgs_ocr = dict()
            for sub_img_idx, sub_img in sub_imgs.items():
                white_pxls = get_pxls(sub_img, [255, 255, 255])
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

            path = 'out/'
            new_f_name = f_name.split('/')[-1]
            write_texts(final_ocr_results, path + new_f_name + '.txt')

            print(f_name)
            print('-------------------------------------------------------')
            print(final_ocr_results)
        except Exception as e:
            print('-------------------------------------------------------')
            print(e)
            print('-------------------------------------------------------')


def write_texts(texts, file):
    with open(file, 'w') as f:
        for idx, text in texts.items():
            f.write(str(idx) + ',' + text + '\n')
    f.close()


def fix_texts():
    """
    fix the text files based on the typical nr of text prompts in the meme, and
    write them back to the original file
    :return: nothing
    """
    # read all the memes from the feather file to validate
    memes = pd.read_feather('data/memes.feather')
    # get counts of the text prompts
    memes['length'] = memes['boxes'].apply(lambda x: len(x))
    # get the most common nr of text prompts
    memes = memes.join(
        memes.groupby('title')['length'].median().rename('median_length'),
        on='title', rsuffix='')
    # read all the text files
    for f_name in tqdm(os.listdir('out')):
        if not f_name.endswith('.txt'):
            continue
        with open('out/' + f_name, 'r') as f:
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
        with open('out/' + f_name, 'w') as f:
            for idx, text in enumerate(ordered_texts):
                f.write(str(idx) + ',' + text + '\n')


def rate_texts():
    """
    read all the text files and rate them based on the text that should be there
    according to the dataset

    we will rate using the levenshtein distance between the actual text and the
    text obtained by OCR
    :return: rating as an inverse sum of levenshtein distances
    """
    # read all the memes from the feather file to validate
    memes = pd.read_feather('data/memes.feather')
    # the distances of the ocr to the real meme texts
    dists = list()
    max_dists = list()
    # read all the text files
    # iterate over all the text files and rate them
    for f_name in tqdm(os.listdir('out')):
        if not f_name.endswith('.txt'):
            continue
        with open('out/' + f_name, 'r') as f:
            lines = f.readlines()
            texts = [line.split(',')[1][:-1] for line in lines]

        # find matching dataset meme
        img_hash = f_name.split('.')[0]
        meme = \
            memes[memes['url'].str.contains(img_hash)]
        text_boxes = meme['boxes'].values[0]
        dist = 0
        max_dist = 0
        for real_text, ocr_text in zip(text_boxes, lines):
            max_dist += max(len(real_text), len(ocr_text))
            if not len(ocr_text):
                dist += len(real_text)
            else:
                dist += distance.levenshtein(real_text, ocr_text)
        dists.append(dist)
        max_dists.append(max_dist)

    print(sum(dists) / sum(max_dists))
    print(sum(dists), sum(max_dists))
    print(max(dists))
    print(min(dists))
    sns.histplot(x=dists, bins=30)
    plt.axvline(np.median(dists), 0, 0.17, color='red')
    plt.axvline(10, 0, 0.17, color='green')
    plt.show()

    sns.boxplot(dists)
    plt.show()


if __name__ == '__main__':
    # show_top_100_ocr()
    # fix_texts()
    rate_texts()
    # imgs = load_images('examples/*.jpg')
    # print(imgs.keys())
    # img = list(imgs.values())[8]
    # img = clean_img(img)
    # show_img(img)
    # sub_imgs = text_box(img)
    # for sub_img_idx, sub_img in sub_imgs.items():
    #     show_img(sub_img)
    # texts = ocr_images(sub_imgs)
    # for key, value in texts.items():
    #     print(key)
    #     print(clean_text(value))
