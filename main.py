# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import copy
import itertools
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import filters
from scipy.signal import find_peaks


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    img1 = cv2.imread('twoellipses.jpg', 0)
    down_width = 256
    down_height = 256
    down_points = (down_width, down_height)
    img1 = cv2.resize(img1, down_points, interpolation=cv2.INTER_LINEAR)

    # plt.subplot(1, 1, 1), plt.imshow(img1, 'gray')
    # plt.title('Original Image')
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    # ret, img1 = cv2.threshold(img1, 100, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(img1, 100, 200)

    # images = [img1, img]
    # titles = ['og', 'canny']
    # for i in range(2):
    #     plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    # plt.show()

    gX = cv2.Sobel(img1, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    gY = cv2.Sobel(img1, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)

    width = canny.shape[0]
    height = canny.shape[1]

    indices = []
    lines = []
    plt.subplot(1, 1, 1), plt.imshow(canny, 'gray')
    plt.title('Canny')
    plt.xticks([]), plt.yticks([])
    plt.show()
    # K = [15, 20, 25, 30, 35, 45]
    k = 30
    while k <= 55:
    # for k in K:
        for i in range(width):
            for j in range(height - k):
                if canny[i][j] != 0 and canny[i][j + k] != 0:  # edge pixels
                    indices.append([[i, j], [i, j + k]])
        k += 1

    # rows = int(height / 10)
    # cols = int(width / 10)

    accumulatorArray = cv2.resize(canny, (0, 0), fx=0.1, fy=0.1)
    # accumulatorArray = copy.deepcopy(canny)
    cols = accumulatorArray.shape[0]
    rows = accumulatorArray.shape[1]
    # for i in range(rows):
    #     col = []
    #     for j in range(cols):
    #         col.append(0)
    #     accumulatorArray.append(col)
    for i in range(cols):
        for j in range(rows):
            accumulatorArray[i][j] = 0
    count = 0
    print(len(accumulatorArray))
    for i in range(len(indices)):
        p1 = indices[i][0]
        p2 = indices[i][1]

        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        m1 = round((x1 + x2)/2)
        m2 = round((y1 + y2)/2)
        XI1 = math.atan(gY[p1[0]][p1[1]] / gX[p1[0]][p1[1]])
        XI2 = math.atan(gY[p2[0]][p2[1]] / gX[p2[0]][p2[1]])
        print('XI1 + XI2 = ')
        print(XI1)
        print(XI2)

        if (XI2 - XI1) == 0:
            count += 1
            print(XI2 - XI1)
        print("THE COUNT =")
        print(count)
        t1 = (y1 - y2 - x1*XI1 + x2*XI2) / (XI2 - XI1) if (XI2 - XI1) else 1
        t2 = (XI1*XI2*(x2 - x1) - y2*XI1 + y1*XI2) / (XI2 - XI1) if (XI2 - XI1) else 1
        m0 = (t2 - m2) / (t1 - m1)
        b0 = (m2*t1 - m1*t2) / (t1 - m1)

        # m = [m1, m2]
        start = [round(t2), round(t1)]
        # x = 10*(cols-1)
        # finish = [int(x * (t2 - m2) / (t1 - m1) + (m2 * t1 - m1 * t2) / (t1 - m1)), x]
        # line = cv2.circle(canny, (y1, x1), radius=5, color=(255, 255, 255),thickness=-1)
        # line = cv2.circle(canny, (y2, x2), radius=5, color=(255, 255, 255),thickness=-1)
        # line = cv2.circle(canny, (m2, m1), radius=5, color=(255, 255, 255), thickness=-1)
        # line = cv2.circle(canny, (round(t2), round(t1)), radius=5, color=(255, 0, 0), thickness=-1)
        # test1 = [p1[1], p1[0]]
        # test2 = [p2[1], p2[0]]
        # line = cv2.line(canny, test1, [m2, m1], (255, 255, 255))
        # line = cv2.line(canny, [m2, m1], test2, (255, 255, 255))

        # line = cv2.line(canny, start, [m2, m1], (255, 255, 255))
        #
        # plt.subplot(1, 1, 1), plt.imshow(line)
        # plt.title('za lines')
        # plt.xticks([]), plt.yticks([])
        # plt.show()

        for i in range(cols):
            x = 10*i
            y = x * (t2 - m2) / (t1 - m1) + (m2 * t1 - m1 * t2) / (t1 - m1)
            y = round(y/10)
            print("first part of line equation = ")
            print((t2 - m2) / (t1 - m1) )
            print("second part = ")
            print((m2 * t1 - m1 * t2) / (t1 - m1))
            print("x = ")
            print(x)
            print("y before = ")
            print(10*y)
            print("i = ")
            print(i)
            print("y = ")
            print(y)
            # line = cv2.circle(canny, (10*y, x), radius=5, color=(255, 255, 255), thickness=-1)
            # line = cv2.line(canny, [m2, m1], [10*y, x], (255, 255, 255))
            #
            # plt.subplot(1, 1, 1), plt.imshow(line)
            # plt.title('za lines')
            # plt.xticks([]), plt.yticks([])
            # plt.show()

            lines.append([(t2 - m2) / (t1 - m1), (m2 * t1 - m1 * t2) / (t1 - m1)])
            if 0 <= y < rows:
                accumulatorArray[i][y] += 1
        # for i in range(256):
        #     for j in range(256):
        #         y = i * (t2 - m2) / (t1 - m1) + (m2 * t1 - m1 * t2) / (t1 - m1)
        #         y = round(y)
        #         if 0 <= y < 256:
        #             accumulatorArray[i][y] += 1


    # threshold = max(max(accumulatorArray))
    threshold = 0
    for i in range(cols):
        for j in range(rows):
            if accumulatorArray[i][j] > threshold:
                threshold = accumulatorArray[i][j]
    print('threshold = ')
    print(threshold)

    centers = []
    for i in range(cols):
    # i = 0
    # j = 0
    # while i < cols:
        for j in range(rows):
        # while j <rows:
            if accumulatorArray[i][j] >= threshold - 20:
                centers.append([i, j])
                print('in centers append')
        #     j += 10
        # i += 10


    # for i in range(len(centers)):
    #     newimage = cv2.circle(img1, (10*centers[i][1], 10*centers[i][0]), radius=5, color=(0, 0, 255), thickness=-1)
    # print(len(indices))
    for i in range(len(indices)):
        newimage = cv2.circle(canny, (indices[i][0][1], indices[i][0][0]), radius=3, color=(255, 255, 255), thickness=-1)
        newimage = cv2.circle(canny, (indices[i][1][1], indices[i][1][0]), radius=3, color=(255, 255, 255), thickness=-1)

    plt.subplot(1, 1, 1), plt.imshow(newimage, 'gray')
    plt.title('all indices')
    plt.xticks([]), plt.yticks([])
    plt.show()
    new = canny
    for i in range(len(centers)):
        new = cv2.circle(canny, [10*centers[i][1], 10*centers[i][0]], radius=10, color=(255, 255, 255), thickness=-1)

    plt.subplot(1, 1, 1), plt.imshow(new, 'gray')
    plt.title('winner center points')
    plt.xticks([]), plt.yticks([])
    plt.show()




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
