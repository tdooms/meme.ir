import cv2
import numpy as np

from color import is_grayscale, is_white, is_black, get_pxl_distance

import swtloc as swt
import numpy as np
from swtloc.configs import (IMAGE_ORIGINAL_MASKED_WORD_LOCALIZATIONS,
                            IMAGE_CONNECTED_COMPONENTS_1C,
                            IMAGE_CONNECTED_COMPONENTS_3C)

from src.ocr.blob import get_blobs, remove_empty_blobs, remove_big_blobs, \
    remove_long_blobs, remove_blob_in_blob


def unify_except_color(img, except_c, unify_c):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.array_equal(img[i, j], except_c):
                continue
            img[i, j] = unify_c
    return img


def swt_transform(img_path):
    # Initializing the SWTLocalizer class with the image path
    swtl = swt.SWTLocalizer(image_paths=img_path)
    swtImgObj = swtl.swtimages[0]
    # Perform Stroke Width Transformation
    swt_mat = swtImgObj.transformImage(text_mode='lb_df',
                                       maximum_angle_deviation=np.pi / 2,
                                       gaussian_blurr_kernel=(5, 5),
                                       minimum_stroke_width=1,
                                       maximum_stroke_width=40,
                                       include_edges_in_swt=False,
                                       display=False,
                                       engine='python')

    # Localizing Letters
    localized_letters = swtImgObj.localizeLetters(minimum_pixels_per_cc=5,
                                                  maximum_pixels_per_cc=1600,
                                                  display=False)

    # Calculate and Draw Words Annotations
    localized_words = swtImgObj.localizeWords(display=False)

    # get the image with valid strokes
    img = \
        swtImgObj._get_image_for_code(IMAGE_CONNECTED_COMPONENTS_3C)[0]
    # convert non-black pixels to white
    img = unify_except_color(img, except_c=[0, 0, 0], unify_c=[255, 255, 255])

    return img


def from_to_color(img, from_color, to_color):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.array_equal(img[i, j], from_color):
                img[i, j] = to_color
    return img


def rotate_img(img, angle):
    # https://pyimagesearch.com/2021/01/20/opencv-rotate-image/ (18-11-2022)
    # grab the dimensions of the image and calculate the center of the image
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # get rotation matrix
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    # rotate image
    return cv2.warpAffine(img, M, (w, h))


def grayscale_img(img, f_name):
    if not is_grayscale(f_name):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def thin_img(img):
    return cv2.ximgproc.thinning(img)


def remove_color(img):
    for row_idx, row in enumerate(img):
        for col_idx, pxl in enumerate(row):
            # if pixel isn't black or white enough then make it fuchsia
            avg_gray = np.average(pxl)
            gray_pxl = [avg_gray for _ in range(3)]
            if is_white(pxl):
                img[row_idx, col_idx] = [255, 255, 255]
            elif is_black(pxl):
                img[row_idx, col_idx] = [0, 0, 0]
            elif avg_gray < 90 and get_pxl_distance(pxl, gray_pxl) < 10:
                img[row_idx, col_idx] = [0, 0, 0]
            elif avg_gray >= 200 and get_pxl_distance(pxl, gray_pxl) < 10:
                img[row_idx, col_idx] = [255, 255, 255]
            else:
                img[row_idx, col_idx] = [255, 0, 255]
    return img


def bilateral_img(img):
    return cv2.bilateralFilter(img, 5, 55, 60)


def clean_img(img):
    # reduce size if width is above 800px
    if img.shape[0] > 800:
        img = cv2.pyrDown(img)

    # smooth image
    img = bilateral_img(img)

    # transform pixel to fuchsia if it is coloured
    img = remove_color(img)
    img = from_to_color(img, [255, 0, 255], [0, 0, 0])

    # remove blobs that are mostly empty
    white_blobs = get_blobs(img, [255, 255, 255])
    black_blobs = get_blobs(img, [0, 0, 0])
    img = remove_empty_blobs(img, white_blobs)
    img = remove_empty_blobs(img, black_blobs)

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


def clean_images(imgs):
    clean_imgs = dict()
    for f_name, img in imgs.items():
        clean_imgs[f_name] = clean_img(img)
    return clean_imgs
