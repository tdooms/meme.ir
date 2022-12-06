import cv2
import numpy as np

from color import is_grayscale, is_white, is_black, get_pxl_distance


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


def to_fuchsia(img):
    for row_idx, row in enumerate(img):
        for col_idx, pxl in enumerate(row):
            # if pixel isn't black or white enough then make it fuchsia
            avg_gray = np.average(pxl)
            gray_pxl = [avg_gray for _ in range(3)]
            if is_white(pxl):
                img[row_idx, col_idx] = [255, 255, 255]
            elif is_black(pxl):
                img[row_idx, col_idx] = [0, 0, 0]
            elif avg_gray < 128 and get_pxl_distance(pxl, gray_pxl) < 12:
                img[row_idx, col_idx] = [0, 0, 0]
            elif avg_gray >= 128 and get_pxl_distance(pxl, gray_pxl) < 12:
                img[row_idx, col_idx] = [255, 255, 255]
            else:
                img[row_idx, col_idx] = [255, 0, 255]
    return img


def bilateral_img(img):
    return cv2.bilateralFilter(img, 5, 55, 60)
