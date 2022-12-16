import scipy.spatial as scs
from PIL import ImageStat, Image


def get_pxls(img, color):
    pxls = list()
    for i, row in enumerate(img):
        for j, pxl in enumerate(row):
            if is_same_pxl(pxl, color):
                pxls.append((i, j))
    return pxls


def is_grayscale(path="image.jpg"):
    im = Image.open(path).convert("RGB")
    stat = ImageStat.Stat(im)
    if sum(stat.sum) / 3 == stat.sum[0]:  # check the avg with any element value
        return True  # if grayscale
    else:
        return False  # else its colour


def get_pxl_distance(pxl1, pxl2):
    return scs.distance.euclidean(pxl1, pxl2)


def is_same_pxl(pxl1, pxl2):
    return pxl1[0] == pxl2[0] and pxl1[1] == pxl2[1] and pxl1[2] == pxl2[2]


def is_black(pxl):
    return scs.distance.euclidean(pxl, [0, 0, 0]) < 10


def is_white(pxl, approx=True):
    if approx:
        return scs.distance.euclidean(pxl, [255, 255, 255]) < 10
    else:
        r, g, b = pxl
        return r == 255 and g == 255 and b == 255


def is_fuchsia(pxl):
    r, g, b = pxl
    return r == 255 and g == 0 and b == 255


def _grayscale_values():
    return [[v, v, v] for v in range(256)]


grayscale_values = _grayscale_values()
