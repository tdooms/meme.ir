import glob
import re

import cv2
import numpy as np
import pytesseract
from PIL import Image
from sklearn.linear_model import LinearRegression

from blob import remove_big_blobs, get_blobs, remove_long_blobs, \
    remove_blob_in_blob
from color import get_pxls
from processing import bilateral_img, from_to_color, to_fuchsia, rotate_img
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
    custom_config = r"--oem 3 --psm 12 -c tessedit_char_whitelist='ABCDEFGHIJKLMNOPQRSTUVWXYZ :' --tessdata-dir /usr/share/tesseract-ocr/5/tessdata/"
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

    return sub_imgs


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
    for f_name, img in imgs.items():
        img = clean_img(img)
        sub_imgs = text_box(img)
        sub_imgs_ocr = dict()
        for sub_img_idx, sub_img in enumerate(sub_imgs):
            white_pxls = get_pxls(sub_img, [255, 255, 255])
            deskew_angle = get_deskew_angle(white_pxls)
            sub_img = rotate_img(sub_img, deskew_angle)
            sub_imgs_ocr[sub_img_idx] = sub_img
        ocr_results = ocr_images(sub_imgs_ocr)
        # clean OCR results


def no_caps_ocr():
    """
    test both small and large letter OCR recognition of tesseract
    :return:
    """
    imgs = load_images('examples/*.jpg')
    new_imgs = dict()
    new_imgs['big_letter.jpg'] = imgs['examples/big_letter.jpg']
    new_imgs['small_letter.jpg'] = imgs['examples/small_letter.jpg']
    # tesseract should return same result for both images
    ocr_results = ocr_images(new_imgs)
    print('real value: ')
    print("THIS IS SOME TEXT TO RECOGNIZE\nWHEREVER THIS GOES IT'S FINE")
    print('----------------------------------------------------------')
    print('big letter result')
    print(ocr_results['big_letter.jpg'])
    print('----------------------------------------------------------')
    print("This is some text to recognize\nwherever this goes it's fine")
    print('----------------------------------------------------------')
    print('small letter result')
    print('----------------------------------------------------------')
    print(ocr_results['small_letter.jpg'])


if __name__ == '__main__':
    no_caps_ocr()
    # imgs = load_images('examples/*.jpg')
    # print(imgs.keys())
    # img = list(imgs.values())[2]
    # img = bilateral_img(img)
    # new_imgs = {'temp': img}
    # texts = ocr_images(new_imgs)
    # print(texts)
    # for key, value in texts.items():
    #     print(key, value)
    #     print(clean_text(value))
    # show_top_100_ocr()
