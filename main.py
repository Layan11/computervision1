# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import itertools
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    img1 = cv2.imread('frame_0_delay-0.2s.jpg', 0)

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
    rows = int(height / 10)
    cols = int(width / 10)
    accumulatorArray = []
    for i in range(rows):
        col = []
        for j in range(cols):
            col.append(0)
        accumulatorArray.append(col)
    plt.subplot(1, 1, 1), plt.imshow(canny, 'gray')
    plt.title('Canny')
    plt.xticks([]), plt.yticks([])
    plt.show()

    for k in range(100, 300):
        for i in range(width):
            for j in range(height - k):
                if canny[i][j] != 0 and canny[i][j + k] != 0:  # edge pixels
                    # print([i, j])
                    # print([i + 200, j])
                    # cv2.circle(canny, ([j, i]), radius=5, color=(255, 255, 255), thickness=-1)
                    # cv2.circle(canny, ([j+200, i]), radius=5, color=(255, 255, 255), thickness=-1)
                    # # cv2.imshow('Test image', canny)
                    # # cv2.waitKey(0)
                    # plt.subplot(1, 1, 1), plt.imshow(canny, 'gray')
                    # plt.title('errrrrrrrrrr')
                    # plt.xticks([]), plt.yticks([])
                    # plt.show()

                    indices.append([[i, j], [i, j + k]])
    # for i in range(tl.shape[0]):
    #     for j in range(tl.shape[1]):
    #         if tl[i][j] != 0:  # edge pixels
    #             indices.append([i, j])
    #
    # print('lennnn')
    # print(len(indices))
    # indices = list(itertools.combinations(indices, 2))
    # print(len(indices))
    # # now we have all possible pairs indices in 'indices'
    ems = []
    for i in range(len(indices)):
        p1 = indices[i][0]
        p2 = indices[i][1]

        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        m1 = int((x1 + x2)/2)
        m2 = int((y1 + y2)/2)
        # ems.append([x1, y1])
        # ems.append([x2, y2])
        # ems.append([m1, m2])
        # for i in range(len(ems)):
        #     newimage = cv2.circle(canny, (ems[i][1], ems[i][0]), radius=5, color=(255, 255, 255),thickness=-1)
        #
        #     plt.subplot(1, 1, 1), plt.imshow(newimage, 'gray')
        #     plt.title('emsssssss')
        #     plt.xticks([]), plt.yticks([])
        #     plt.show()
        XI1 = math.atan2(gY[p1[0]][p1[1]], gX[p1[0]][p1[1]])
        XI2 = math.atan2(gY[p2[0]][p2[1]], gX[p2[0]][p2[1]])
        t1 = (y1 - y2 - x1*XI1 + x2*XI2) / (XI2 - XI1) if (XI2 - XI1) else 1
        t2 = (XI1*XI2*(x2 - x1) - y2*XI1 + y1*XI2) / (XI2 - XI1) if (XI2 - XI1) else 1
        m0 = (t2 - m2) / (t1 - m1)
        b0 = (m2*t1 - m1*t2) / (t1 - m1)

        for i in range(rows):
            x = 10*i
            y = x * (t2 - m2) / (t1 - m1) + (m2 * t1 - m1 * t2) / (t1 - m1)
            y = int(y/10)

            if 0 <= y < cols:
                accumulatorArray[i][y] += 1

    threshold = max(max(accumulatorArray))
    centers = []
    for i in range(rows):
        for j in range(cols):
            if accumulatorArray[i][j] > threshold:
                centers.append([i, j])


    # for i in range(len(centers)):
    #     newimage = cv2.circle(img1, (10*centers[i][1], 10*centers[i][0]), radius=5, color=(0, 0, 255), thickness=-1)
    print(len(indices))
    for i in range(len(indices)):
        newimage = cv2.circle(canny, (indices[i][0][1], indices[i][0][0]), radius=2, color=(255, 255, 255), thickness=-1)
        newimage = cv2.circle(canny, (indices[i][1][1], indices[i][1][0]), radius=2, color=(255, 255, 255), thickness=-1)

    plt.subplot(1, 1, 1), plt.imshow(newimage, 'gray')
    plt.title('all indices')
    plt.xticks([]), plt.yticks([])
    plt.show()
    new = canny
    for i in range(len(centers)):
        new = cv2.circle(canny, [10*centers[i][1], 10*centers[i][0]], radius=5, color=(255, 255, 255), thickness=-1)

    plt.subplot(1, 1, 1), plt.imshow(new, 'gray')
    plt.title('winner center points')
    plt.xticks([]), plt.yticks([])
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
