# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import itertools
import math

import cv2
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    img1 = cv2.imread('alvtd333_alvin_template_small_ellipse.png', 0)
    cv2.imshow("im1", img1)
    cv2.waitKey(0)

    # ret, img1 = cv2.threshold(img1, 100, 255, cv2.THRESH_BINARY)

    img = cv2.Canny(img1, 100, 200)
    cv2.imshow('Canny', img)
    cv2.waitKey(0)


    gX = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    gY = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)

    # the gradient magnitude images are now of the floating point data
    # type, so we need to take care to convert them back a to unsigned
    # 8-bit integer representation so other OpenCV functions can operate
    # on them and visualize them
    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)
    # combine the gradient representations into a single image
    combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
    # show our output images
    cv2.imshow("Sobel/Scharr X", gX)
    cv2.imshow("Sobel/Scharr Y", gY)
    cv2.imshow("Sobel/Scharr Combined", combined)
    cv2.waitKey(0)

    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
        for j in range(width):
            img[i][j] = math.atan2(gX[i][j], gY[i][j])

    # finding the center of the ellipse
    votes = []
    indices = []
    for i in range(height):
        for j in range(width):
            votes.append(0)
    print(math.atan2(0, 2))
    print(math.atan2(2, 0))
    for i in range(height):
        for j in range(width):
            if img[i][j] != 0:
                indices.append([i, j])

    # indices = list(itertools.combinations(indices, 2))
    # print(len(indices))
    # now we have all possible pairs indices in 'indices'
    # for i in range(len(indices)):
    #     p1 = indices[i][0]
    #     p2 = indices[i][1]
    #     x1, y1 = p1[0], p1[1]


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
