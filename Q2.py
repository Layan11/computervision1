import math
import cv2
import numpy

if __name__ == '__main__':

    target = cv2.imread('target.jpg')
    im1 = cv2.imread('source_01.jpg')

    sift1 = cv2.SIFT_create()
    sift2 = cv2.SIFT_create()

    keypoints1, descriptors1 = sift1.detectAndCompute(target, None)
    keypoints2, descriptors2 = sift2.detectAndCompute(im1, None)

    n, m = len(keypoints1), len(keypoints2)

    matrix = []
    matches = []
    for i in range(n):
        cols = []
        for j in range(m):
            cols.append(numpy.linalg.norm(descriptors1[i] - descriptors2[j]))
        matrix.append(cols)

        sortedvalues = cols
        sortedvalues.sort()
        first_min = sortedvalues[0]
        second_min = sortedvalues[1]
        idx = cols.index(first_min)
        if first_min / second_min < 0.8:
            matches.append(idx)
    # try and print
    for i in range(len(matches)):
        x = keypoints2[matches[i]].pt[0] + target.size[0]
        y = keypoints2[matches[i]].pt[1]
        matchimg = cv2.line(matchimg, (round(keypoints1[i].pt[0]), round(keypoints1[i].pt[1])), (round(x), round(y)),
                                       (255, 225, 128), 1)
    cv2.imshow("image", matchimg)
    cv2.waitKey(0)

    # now what to do with matches??
