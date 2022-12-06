import copy

import numpy as np
from tqdm import tqdm

from color import is_same_pxl


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
    ih, iw = img.shape[:2]
    for blob in blobs:
        minx_blob = min(blob, key=lambda x: x[1])[1]
        maxx_blob = max(blob, key=lambda x: x[1])[1]
        width = maxx_blob - minx_blob

        miny_blob = min(blob, key=lambda x: x[0])[0]
        maxy_blob = max(blob, key=lambda x: x[0])[0]
        height = maxy_blob - miny_blob

        if height < 5:
            blobs_to_remove.extend(blob)
            pass
        elif height != 0 and width / height > 2:
            blobs_to_remove.extend(blob)
        elif height / ih > 0.15 or width / iw > 0.15 and height / width > 2:
            blobs_to_remove.extend(blob)
        elif height / iw < 0.01:
            blobs_to_remove.extend(blob)

    for bloby, blobx in blobs_to_remove:
        img[bloby, blobx] = [255, 0, 255]

    return img


# floodfill algorithm
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


# Function that returns true if the given pixel is valid
def is_valid(img, h, w, row_idx, col_idx, prevC, newC):
    if row_idx < 0 or row_idx >= h \
            or col_idx < 0 or col_idx >= w or \
            is_same_pxl(img[row_idx][col_idx], newC) or \
            not is_same_pxl(img[row_idx][col_idx], prevC):
        return False
    return True
