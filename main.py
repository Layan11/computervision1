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
    img1 = cv2.imread('alvtd333_alvin_template_small_ellipse.png', 0)
    titles = ['image']
    images = [img1]
    for i in range(1):
        plt.subplot(1, 1, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    # cv2.imshow("im1", img1)
    # cv2.waitKey(0)


    # ret, img1 = cv2.threshold(img1, 100, 255, cv2.THRESH_BINARY)

    img = cv2.Canny(img1, 100, 200)
    # cv2.imshow('Canny', img)
    # cv2.waitKey(0)


    gX = cv2.Sobel(img1, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    gY = cv2.Sobel(img1, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    sobelx = np.uint8(np.absolute(gX))
    sobely = np.uint8(np.absolute(gY))
    cv2.imshow('sobelx', sobelx)
    cv2.waitKey(0)
    cv2.imshow('sobely', sobely)
    cv2.waitKey(0)
    # the gradient magnitude images are now of the floating point data
    # type, so we need to take care to convert them back a to unsigned
    # 8-bit integer representation so other OpenCV functions can operate
    # on them and visualize them
    # gX = cv2.convertScaleAbs(gX)
    # gY = cv2.convertScaleAbs(gY)
    # combine the gradient representations into a single image
    # combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
    # show our output images
    # cv2.imshow("Sobel/Scharr X", gX)
    # cv2.imshow("Sobel/Scharr Y", gY)
    # cv2.imshow("Sobel/Scharr Combined", combined)
    # cv2.waitKey(0)

    height = img.shape[0]
    width = img.shape[1]

    # (cX, cY) = (width // 2, height // 2)
    # # since we are using NumPy arrays, we can apply array slicing to grab
    # # large chunks/regions of interest from the image -- here we grab the
    # # top-left corner of the image
    # tl = img[110:cY-111, 100:cX]
    # plt.subplot(1, 1, 1), plt.imshow(tl, 'gray')
    # plt.title('tl')
    # plt.xticks([]), plt.yticks([])
    # plt.show()


    # for i in range(height):
    #     for j in range(width):
    #         img[i][j] = math.atan2(gX[i][j], gY[i][j])

    # finding the center of the ellipse

    indices = []
    for i in range(height - 20):
        for j in range(width):
            if img[i][j] != 0 and img[i + 20][j] != 0:  # edge pixels
                indices.append([[i, j], [i+20, j]])
    # for i in range(tl.shape[0]):
    #     for j in range(tl.shape[1]):
    #         if tl[i][j] != 0:  # edge pixels
    #             indices.append([i, j])
    #
    print(len(indices))
    # indices = list(itertools.combinations(indices, 2))
    # print(len(indices))
    # # now we have all possible pairs indices in 'indices'
    for i in range(len(indices)):
        p1 = indices[i][0]
        p2 = indices[i][1]

        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        m1 = (x1 + x2)/2
        m2 = (y1 + y2)/2
        # XI1 = img[p1[0]][p1[1]]
        XI1 = math.atan2(gY[p1[0]][p1[1]], gX[p1[0]][p1[1]])
        # XI2 = img[p2[0]][p2[1]]
        XI2 = math.atan2(gY[p2[0]][p2[1]], gX[p2[0]][p2[1]])
        print(XI1)
        print(XI2)
        t1 = (y1 - y2 - x1*XI1 + x2*XI2) / (XI2 - XI1) if (XI2 - XI1) else 1
        t2 = (XI1*XI2*(x2 - x1) - y2*XI1 + y1*XI2) / (XI2 - XI1) if (XI2 - XI1) else 1
        print(t1)
        print(t2)
        m0 = (t2 - m2) / (t1 - m1)
        b0 = (m2*t1 - m1*t2) / (t1 - m1)

        # print(m0)
        # print(b0)
        # votes[m0][b0] += 1
        cols = int(height/10)
        rows = int(width/10)
        accumulatorArray = []
        for i in range(rows):
            col = []
            for j in range(cols):
                col.append(0)
            accumulatorArray.append(col)
        # accumulatorArray = arr = [[0]*cols]*rows
        for i in range(rows):
            x = 10*i
            y = x * (t2 - m2) / (t1 - m1) + (m2 * t1 - m1 * t2) / (t1 - m1)
            y = int(y/10)
            if y < cols:
                accumulatorArray[i][y] += 1
        print(accumulatorArray)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
