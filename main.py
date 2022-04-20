# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import copy
import math
import cv2
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from scipy.ndimage import filters


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    images = ['one.jpg', '1271488188_2077d21f46_b.jpg', 'images.jpg', 'alvtd333_alvin_template_small_ellipse.png',
              's-l400.jpg', 'gettyimages-1212455495-612x612.jpg', 'nEKGD2wNiwqrTOc63kiWZT7b4.png']
    kmin = [34, 13, 10, 10, 40, 45, 34]
    kmax = [48, 55, 50, 50, 100, 80, 48]
    step = [1, 10, 4, 1, 5, 4, 2]
    maxneigh = [20, 4, 7, 5, 7, 14, 7]
    minneigh = [20, 3, 7, 4, 7, 14, 7]
    thresholds = [21, 33, 60, 110, 235, 13, 60]
    sigma = [2.03, -1, -1, -1, 2, 2.2, 2.03]
    cannylow = [100, 100, 100, 100, 65, 100, 100]
    cannyhigh = [200, 200, 200, 200, 70, 200, 200]
    centerradius = [8, 6, 8, 4, 8, 8, 8]

    for idx in range(len(images)):
        grey = cv2.imread(images[idx], 0)
        color = cv2.imread(images[idx])
        down_width = 256
        down_height = 256
        down_points = (down_width, down_height)
        grey = cv2.resize(grey, down_points, interpolation=cv2.INTER_LINEAR)
        color = cv2.resize(color, down_points, interpolation=cv2.INTER_LINEAR)
        color2 = cv2.resize(color, down_points, interpolation=cv2.INTER_LINEAR)
        if sigma[idx] != -1:
            grey = scipy.ndimage.gaussian_filter(grey, sigma[idx])
        canny = cv2.Canny(grey, cannylow[idx], cannyhigh[idx])

        gX = cv2.Sobel(grey, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
        gY = cv2.Sobel(grey, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)

        width = canny.shape[0]
        height = canny.shape[1]

        indices = []
        lines = []

        k = kmin[idx]
        while k <= kmax[idx]:
            for i in range(width):
                for j in range(height - k):
                    if canny[i][j] != 0 and canny[i][j + k] != 0:  # edge pixels
                        indices.append([[i, j], [i, j + k]])
            k += step[idx]


        accumulatorArray = cv2.resize(canny, (0, 0), fx=0.1, fy=0.1)
        cols = accumulatorArray.shape[0]
        rows = accumulatorArray.shape[1]

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

            t1 = (y1 - y2 - x1*XI1 + x2*XI2) / (XI2 - XI1) if (XI2 - XI1) else 1
            t2 = (XI1*XI2*(x2 - x1) - y2*XI1 + y1*XI2) / (XI2 - XI1) if (XI2 - XI1) else 1
            m0 = (t2 - m2) / (t1 - m1)
            b0 = (m2*t1 - m1*t2) / (t1 - m1)
            start = [round(t2), round(t1)]

            for i in range(cols):
                x = 10*i
                y = x * (t2 - m2) / (t1 - m1) + (m2 * t1 - m1 * t2) / (t1 - m1)
                y = round(y/10)

                lines.append([(t2 - m2) / (t1 - m1), (m2 * t1 - m1 * t2) / (t1 - m1)])
                if 0 <= y < rows:
                    accumulatorArray[i][y] += 1

        threshold = 0
        for i in range(cols):
            for j in range(rows):
                if accumulatorArray[i][j] > threshold:
                    threshold = accumulatorArray[i][j]

        centers = []
        accumulatorArray2 = copy.deepcopy(accumulatorArray)
        data_max = filters.maximum_filter(accumulatorArray2, maxneigh[idx])
        maxima = (accumulatorArray2 == data_max)
        data_min = filters.minimum_filter(accumulatorArray2, minneigh[idx])
        diff = ((data_max - data_min) > thresholds[idx])
        maxima[diff == 0] = 0

        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        x, y = [], []
        for dy, dx in slices:
            x_center = (dx.start + dx.stop - 1) / 2
            x.append(x_center)
            y_center = (dy.start + dy.stop - 1) / 2
            y.append(y_center)

        for i in range(len(x)):
            centers.append([x[i], y[i]])

        # for i in range(len(indices)):
        #     newimage = cv2.circle(canny, (indices[i][0][1], indices[i][0][0]), radius=3, color=(255, 255, 255), thickness=-1)
        #     newimage = cv2.circle(canny, (indices[i][1][1], indices[i][1][0]), radius=3, color=(255, 255, 255), thickness=-1)
        #
        # plt.subplot(1, 1, 1), plt.imshow(newimage, 'gray')
        # plt.title('all indices')
        # plt.xticks([]), plt.yticks([])
        # plt.show()

        result = color

        for i in range(len(centers)):
            result = cv2.circle(result, [int(10 * centers[i][0]), int(10 * centers[i][1])], radius=centerradius[idx],
                                color=(255, 0, 0), thickness=-1)

        resimages = [color2, result]
        titles = ['Image', 'Centers found']
        for i in range(2):
            plt.subplot(1, 2, i + 1), plt.imshow(resimages[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
