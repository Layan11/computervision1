import copy
import math

import cv as cv
import cv2
import numpy
import random

import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    target = cv2.imread('target.jpg')
    # cameleon_set = ['source_01.jpg', 'source_02.jpg', 'source_03.jpg', 'source_04.jpg', 'source_05.jpg', 'source_06.jpg',
    #           'source_07.jpg']
    cameleon_set = ['source_01.jpg', 'source_02.jpg']
    warped_images = []
    for image in cameleon_set:
        im1 = cv2.imread(image)
        # target = cv2.resize(target, (500, 500), interpolation=cv2.INTER_LINEAR)
        # im1 = cv2.resize(im1, (500, 500), interpolation=cv2.INTER_LINEAR)
        sift1 = cv2.SIFT_create()
        sift2 = cv2.SIFT_create()
        print(image)
        # cv2.imshow('imageee', im1)
        # cv2.waitKey(0)
        keypoints1, descriptors1 = sift1.detectAndCompute(target, None)
        keypoints2, descriptors2 = sift2.detectAndCompute(im1, None)

        firsttime = True
        zedge = []
        n, m = len(keypoints1), len(keypoints2)
        desc1_sample = random.sample(range(0, n), round(n/8))
        desc2_sample = random.sample(range(0, m), round(m/8))
        try1 = descriptors1[desc1_sample]
        try2 = descriptors2[desc2_sample]
        matrix = []
        matches = []
        for i in range(round(n/8)):
            cols = []
            for j in range(round(m/8)):
                d = numpy.linalg.norm(descriptors1[i] - descriptors2[j])
                cols.append(d)
            matrix.append(cols)

            sortedvalues = copy.deepcopy(cols)
            sortedvalues.sort()
            first_min = sortedvalues[0]
            second_min = sortedvalues[1]
            idx = cols.index(first_min)

            if first_min / second_min < 0.8:
                zedge.append([i, j])
                matches.append(idx)

            else:
                matches.append(-1)

        h1, w1, ch1 = target.shape
        h2, w2, ch2 = im1.shape
        # # shouldnt they be from matches array tho...???!!!
        # pts1 = np.float32([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]]).reshape(-1, 1, 2)
        # pts2 = np.float32([[0, 0], [0, h2 - 1], [w2 - 1, h2 - 1], [w2 - 1, 0]]).reshape(-1, 1, 2)
        #
        # M = cv2.getPerspectiveTransform(pts1, pts2)
        #
        #
        # dst = cv2.warpPerspective(target, M, (w1, h1), flags = cv2.INTER_CUBIC)
        # cv2.imshow("image", dst)
        # cv2.waitKey(0)

        # matchimg = cv2.hconcat([target, im1])
        points = []
        max_votes = 0
        # matchimg2 = cv2.hconcat([target, im1])
        for i in range(len(matches)):
            if matches[i] != -1:
                x = keypoints2[matches[i]].pt[0] + im1.shape[0]
                y = keypoints2[matches[i]].pt[1]
                # print('the draw')
                # matchimg = cv2.line(matchimg, (round(keypoints1[i].pt[0]), round(keypoints1[i].pt[1])), (round(x), round(y)),
                #                         (255, 225, 128), 1)

                points.append([keypoints1[i].pt, keypoints2[matches[i]].pt])

        # cv2.imshow("image", matchimg)
        # cv2.waitKey(0)

        for i in range(1000):
            p1 = random.choice(points)
            p2 = random.choice(points)
            p3 = random.choice(points)
            p4 = random.choice(points)

            pset1 = numpy.float32([p1[0], p2[0], p3[0], p4[0]])
            pset2 = numpy.float32([[p1[1], p2[1], p3[1], p4[1]]])

            h, status = cv2.findHomography(pset1, pset2)

            projective_matrix = h
            # projective_matrix = cv2.getPerspectiveTransform(pset1, pset2)

            votescount = 0
            for j in range(len(points)):
                currPoint = np.array(points[j])
                currPoint2 = [currPoint[0][0], currPoint[0][1], 1]

                new_proj_point = np.matmul(projective_matrix, currPoint2)
                new_proj_point = [(new_proj_point[0] / new_proj_point[2]), (new_proj_point[1] / new_proj_point[2])]
                pixel_distance_proj = numpy.linalg.norm(currPoint[1] - new_proj_point)
                if pixel_distance_proj <= 2:
                    votescount += 1
            if votescount > max_votes:
                max_proj_matrix = projective_matrix
                max_votes = votescount
        # cv2.waitKey(0)
        img2 = cv2.warpPerspective(im1, max_proj_matrix, (w1, h1), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_TRANSPARENT)
        # cv2.imshow("the proj image", img2)
        # cv2.waitKey(0)
        warped_images.append(img2)
    clean_image = np.zeros((h1, w1, ch1))
    cv2.imshow("the proj", warped_images[0] + warped_images[1])

    cv2.imshow("the prxoj", warped_images[1])
    cv2.waitKey(0)
    for im in warped_images:

        clean_image += im
    # clean_image = np.true_divide(clean_image, 2)
    cv2.imshow('CLEAN IMAGE', clean_image)
    cv2.waitKey(0)

