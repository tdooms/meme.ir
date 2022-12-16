import cv2
import numpy as np
from sklearn.linear_model import LinearRegression


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


def text_box(img):
    # Convert the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    # Specify structure shape and kernel size.
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
