import glob
import re

import cv2
import numpy as np
import pytesseract
from PIL import Image

from blob import remove_big_blobs, get_blobs, remove_long_blobs, \
    remove_blob_in_blob
from processing import bilateral_img, from_to_color, to_fuchsia, rotate_img


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
        img = bilateral_img(img)
        img = to_fuchsia(img)
        img = remove_big_blobs(img, [0, 0, 0])
        img = remove_big_blobs(img, [255, 255, 255])
        img = from_to_color(img, [0, 0, 0], [255, 255, 255])
        img = from_to_color(img, [255, 0, 255], [0, 0, 0])
        # img = greyscale_img(img, f_name)
        # img = thin_img(img)
        show_img(img)

        _, img_dark = cv2.threshold(img, 250, 255, 1)
        _, img_light = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
        # show_img(img_light)
        # show_img(img_dark)

        # rotate to find more text
        # img_clock = rotate_img(img, 15)
        # img_counter = rotate_img(img, -15)
        img_clock_d = rotate_img(img_dark, 15)
        img_counter_d = rotate_img(img_dark, -15)
        img_clock_l = rotate_img(img_light, 15)
        img_counter_l = rotate_img(img_light, -15)

        # img_thin_d = cv2.ximgproc.thinning(img_dark)
        # img_thin_l = cv2.ximgproc.thinning(img_light)
        # show_img(img_thin_d)
        # img_thin_d = cv2.erode(img_dark, thinning_kernel, iterations=3)
        # img_thin_l = cv2.erode(img_light, thinning_kernel, iterations=3)

        clean_imgs[f_name] = dict()
        clean_imgs[f_name]['original'] = img
        # clean_imgs[f_name]['clock'] = img_clock
        # clean_imgs[f_name]['counter'] = img_counter
        clean_imgs[f_name]['dark'] = img_dark
        clean_imgs[f_name]['light'] = img_light
        clean_imgs[f_name]['clock_d'] = img_clock_d
        clean_imgs[f_name]['counter_d'] = img_counter_d
        clean_imgs[f_name]['clock_l'] = img_clock_l
        clean_imgs[f_name]['counter_l'] = img_counter_l
        # clean_imgs[f_name]['thin_d'] = img_thin_d
        # clean_imgs[f_name]['thin_l'] = img_thin_l
    return clean_imgs


def ocr_images(imgs):
    custom_config = r"--oem 3 --psm 12 -c tessedit_char_whitelist= 'ABCDEFGHIJKLMNOPQRSTUVWXYZ :'"
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
    img = to_fuchsia(img)
    img = from_to_color(img, [255, 0, 255], [0, 0, 0])

    # removes big nonsensical blobs
    white_blobs = get_blobs(img, [255, 255, 255])
    black_blobs = get_blobs(img, [0, 0, 0])
    img = remove_big_blobs(img, black_blobs)
    img = remove_big_blobs(img, white_blobs)

    # remove long blobs
    white_blobs = get_blobs(img, [255, 255, 255])
    black_blobs = get_blobs(img, [0, 0, 0])
    img = remove_long_blobs(img, black_blobs)
    img = remove_long_blobs(img, white_blobs)

    # remove the blobs within blobs
    white_blobs = get_blobs(img, [255, 255, 255])
    black_blobs = get_blobs(img, [0, 0, 0])
    img = remove_blob_in_blob(img, black_blobs + white_blobs)

    img = from_to_color(img, [0, 0, 0], [255, 255, 255])
    img = from_to_color(img, [255, 0, 255], [0, 0, 0])
    return img


if __name__ == '__main__':
    imgs = load_images('examples/*.jpg')
    print(imgs.keys())
    f_name, img = list(imgs.items())[1]
    clean_img(img)
    # clean_imgs = clean_images(imgs)
    # for f_name, img_dict in clean_imgs.items():
    #     img_texts = ocr_images(img_dict)
    #     print(img_texts)
    #     break
