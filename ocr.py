import copy
from helper import grayscale_values, closest_node
import glob

import imutils
import scipy.spatial as scs

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageStat
from tqdm import tqdm


def load_images(location):
    imgs = dict()
    for file_location in glob.glob(location):
        img = np.array(Image.open(file_location))
        imgs[file_location] = img

    return imgs


def is_grayscale(path="image.jpg"):
    im = Image.open(path).convert("RGB")
    stat = ImageStat.Stat(im)
    if sum(stat.sum) / 3 == stat.sum[0]:  # check the avg with any element value
        return True  # if grayscale
    else:
        return False  # else its colour


def rotate_img(img, angle):
    # https://pyimagesearch.com/2021/01/20/opencv-rotate-image/ (18-11-2022)
    # grab the dimensions of the image and calculate the center of the image
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # get rotation matrix
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    # rotate image
    return cv2.warpAffine(img, M, (w, h))


def greyscale_img(img, f_name):
    if not is_grayscale(f_name):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def thin_img(img):
    return cv2.ximgproc.thinning(img)
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return img


def bilateral_img(img):
    return cv2.bilateralFilter(img, 5, 55, 60)


def show_img(img):
    im = Image.fromarray(img)
    im.show()


def from_to_color(img, from_color, to_color):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.array_equal(img[i, j], from_color):
                img[i, j] = to_color
    return img


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


def is_black(pxl):
    return scs.distance.euclidean(pxl, [10, 10, 10]) < 75


def is_white(pxl, approx=True):
    if approx:
        return scs.distance.euclidean(pxl, [245, 245, 245]) < 70
    else:
        r, g, b = pxl
        return r == 255 and g == 255 and b == 255


def is_fuchsia(pxl):
    r, g, b = pxl
    return r == 255 and g == 0 and b == 255


def is_blob_in_blob(blob1, blob2):
    # search minimal values of blob 1
    ymin_1 = min(blob1, key=lambda coord: coord[0])[0]
    xmin_1 = min(blob1, key=lambda coord: coord[1])[1]
    # search minimal values of blob 2
    ymin_2 = min(blob2, key=lambda coord: coord[0])[0]
    xmin_2 = min(blob2, key=lambda coord: coord[1])[1]

    # search maximal values of blob 1
    xmax_1 = max(blob1, key=lambda coord: coord[1])[1]
    ymax_1 = max(blob1, key=lambda coord: coord[0])[0]
    # search maximal values of blob 2
    xmax_2 = max(blob2, key=lambda coord: coord[1])[1]
    ymax_2 = max(blob2, key=lambda coord: coord[0])[0]

    # check if blob 2 within blob 1
    if not xmax_1 > xmin_2 > xmin_1:
        return False
    if not xmin_1 < xmax_2 < xmax_1:
        return False
    if not ymax_1 > ymin_2 > ymin_1:
        return False
    if not ymin_1 < ymax_2 < ymax_1:
        return False

    # blob within blob
    return True


def remove_blob_in_blob(img, blobs):
    blobs_to_remove_idx = set()
    # pairwise check if blob is in blob
    for i in range(len(blobs)):
        for j in range(len(blobs)):
            if i == j:
                continue

            if is_blob_in_blob(blobs[i], blobs[j]):
                blobs_to_remove_idx.add(j)

    # add blob coordinates
    blobs_to_remove = list()
    for blobs_idx in blobs_to_remove_idx:
        blobs_to_remove.extend(blobs[blobs_idx])
    # remove blobs
    for bloby, blobx in blobs_to_remove:
        img[bloby, blobx] = [255, 0, 255]

    return img


def to_fuchsia(img):
    for row_idx, row in enumerate(img):
        for col_idx, pxl in enumerate(row):
            # if pixel isn't black or white enough then make it fuchsia
            if is_white(pxl):
                img[row_idx, col_idx] = [255, 255, 255]
            elif is_black(pxl):
                img[row_idx, col_idx] = [0, 0, 0]
            # elif closest_node(pxl, grayscale_values) < 2:
            #     img[row_idx, col_idx] = [0, 0, 0]
            else:
                img[row_idx, col_idx] = [255, 0, 255]
    return img


def is_same_pxl(pxl1, pxl2):
    return pxl1[0] == pxl2[0] and pxl1[1] == pxl2[1] and pxl1[2] == pxl2[2]


# Function that returns true if the given pixel is valid
def is_valid(img, h, w, row_idx, col_idx, prevC, newC):
    if row_idx < 0 or row_idx >= h \
            or col_idx < 0 or col_idx >= w or \
            is_same_pxl(img[row_idx][col_idx], newC) or \
            not is_same_pxl(img[row_idx][col_idx], prevC):
        return False
    return True


# FloodFill function
def create_blob(img, h, w, row_idx, col_idx, prevC, newC):
    blob = list()
    queue = list()

    # Append the position of starting pixel of the component
    queue.append([row_idx, col_idx])
    blob.append([row_idx, col_idx])

    # Color the pixel with the new color
    img[row_idx][col_idx] = newC

    # While the queue is not empty i.e. the whole component having prevC color
    # is not colored with newC color
    while queue:

        # Dequeue the front node
        currPixel = queue.pop()

        posX = currPixel[0]
        posY = currPixel[1]

        # Check if the adjacent and diagnonal pixels are valid
        if is_valid(img, h, w, posX + 1, posY, prevC, newC):
            # Color with newC
            # if valid and enqueue
            img[posX + 1][posY] = newC
            queue.append([posX + 1, posY])
            blob.append([posX + 1, posY])

        if is_valid(img, h, w, posX - 1, posY, prevC, newC):
            img[posX - 1][posY] = newC
            queue.append([posX - 1, posY])
            blob.append([posX - 1, posY])

        if is_valid(img, h, w, posX, posY + 1, prevC, newC):
            img[posX][posY + 1] = newC
            queue.append([posX, posY + 1])
            blob.append([posX, posY + 1])

        if is_valid(img, h, w, posX, posY - 1, prevC, newC):
            img[posX][posY - 1] = newC
            queue.append([posX, posY - 1])
            blob.append([posX, posY - 1])

        # diagonal pixels
        if is_valid(img, h, w, posX - 1, posY - 1, prevC, newC):
            img[posX - 1][posY - 1] = newC
            queue.append([posX - 1, posY - 1])
            blob.append([posX - 1, posY - 1])

        if is_valid(img, h, w, posX + 1, posY + 1, prevC, newC):
            img[posX + 1][posY + 1] = newC
            queue.append([posX + 1, posY + 1])
            blob.append([posX + 1, posY + 1])
        if is_valid(img, h, w, posX - 1, posY + 1, prevC, newC):
            img[posX - 1][posY + 1] = newC
            queue.append([posX - 1, posY + 1])
            blob.append([posX - 1, posY + 1])
        if is_valid(img, h, w, posX + 1, posY - 1, prevC, newC):
            img[posX + 1][posY - 1] = newC
            queue.append([posX + 1, posY - 1])
            blob.append([posX + 1, posY - 1])
    return blob


def remove_big_blobs(img, blobs):
    blobs_to_remove = list()
    blobs_too_big = list()
    # add small blobs
    for blob in blobs:
        if len(blob) > 10:
            blobs_too_big.append(blob)
            continue
        blobs_to_remove.append(blob)
    for blob in blobs_too_big:
        if len(blob) < 1700:
            continue
        blobs_to_remove.append(blob)
    # remove blobs if bl
    for blob_to_remove in blobs_to_remove:
        for blob_row, blob_col in blob_to_remove:
            img[blob_row][blob_col] = [255, 0, 255]

    return img


def get_blobs(img, blob_color):
    blobs = list()
    checked_pxls = np.ndarray(shape=img.shape[:2], dtype=bool)
    checked_pxls.fill(False)
    h, w = img.shape[:2]
    for row_idx, row in tqdm(enumerate(img)):
        for col_idx, pxl in enumerate(row):
            if checked_pxls[row_idx][col_idx]:
                continue
            elif not is_same_pxl(pxl, blob_color):
                checked_pxls[row_idx][col_idx] = True
                continue
            # create the blob
            _img = copy.deepcopy(img)
            blob_idces = create_blob(_img, h, w, row_idx, col_idx,
                                     blob_color, [2, 0, 0])
            for blob_idx in blob_idces:
                blob_row_idx = blob_idx[0]
                blob_col_idx = blob_idx[1]
                checked_pxls[blob_row_idx][blob_col_idx] = True
            blobs.append(blob_idces)
    return blobs


def remove_long_blobs(img, blobs):
    blobs_to_remove = list()
    for blob in blobs:
        minx_blob = min(blob, key=lambda x: x[1])[1]
        maxx_blob = max(blob, key=lambda x: x[1])[1]
        width = maxx_blob - minx_blob

        miny_blob = min(blob, key=lambda x: x[0])[0]
        maxy_blob = max(blob, key=lambda x: x[0])[0]
        height = maxy_blob - miny_blob

        if width == 0 or height == 0:
            blobs_to_remove.extend(blob)
        elif width / height > 2:
            blobs_to_remove.extend(blob)
        elif height > 50 or width > 50 and 2 < height / width:
            blobs_to_remove.extend(blob)

    for bloby, blobx in blobs_to_remove:
        img[bloby, blobx] = [255, 0, 255]

    return img


def get_text_areas(img, f_name):
    img = bilateral_img(img)
    # transform pixel to fuchsia if it is coloured
    img = to_fuchsia(img)
    # removes big nonsensical blobs
    white_blobs = get_blobs(img, [255, 255, 255])
    black_blobs = get_blobs(img, [0, 0, 0])
    img = remove_big_blobs(img, black_blobs)
    img = remove_big_blobs(img, white_blobs)
    show_img(img)
    # remove long blobs
    white_blobs = get_blobs(img, [255, 255, 255])
    black_blobs = get_blobs(img, [0, 0, 0])
    img = remove_long_blobs(img, black_blobs)
    img = remove_long_blobs(img, white_blobs)
    show_img(img)
    # remove the blobs within blobs
    white_blobs = get_blobs(img, [255, 255, 255])
    black_blobs = get_blobs(img, [0, 0, 0])
    img = remove_blob_in_blob(img, black_blobs + white_blobs)
    show_img(img)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    # img = cv2.dilate(img, kernel, iterations=1)
    img = from_to_color(img, [0, 0, 0], [255, 255, 255])
    img = from_to_color(img, [255, 0, 255], [0, 0, 0])
    show_img(img)

    # img = greyscale_img(img, f_name)
    #
    # _, img_dark = cv2.threshold(img, 240, 255, 1)
    # _, img_light = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)
    # # show_img(img_dark)
    # # show_img(img_light)
    #
    # # dilate image
    # img_dark = cv2.dilate(img_dark, kernel, iterations=1)
    # show_img(img_light)
    # show_img(img_dark)


if __name__ == '__main__':
    imgs = load_images('examples/*.jpg')
    f_name, img = list(imgs.items())[2]
    get_text_areas(img, f_name)
    # clean_imgs = clean_images(imgs)
    # for f_name, img_dict in clean_imgs.items():
    #     img_texts = ocr_images(img_dict)
    #     print(img_texts)
    #     break
