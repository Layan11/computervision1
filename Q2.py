import cv as cv
import cv2
import numpy as np


if __name__ == '__main__':

    target = cv2.imread('target.jpg')
    gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    cameleon_set = ['source_01.jpg', 'source_02.jpg', 'source_03.jpg', 'source_04.jpg', 'source_05.jpg', 'source_06.jpg',
               'source_07.jpg']
    tshape = target.shape
    averageim = np.zeros(tshape, float)
    num_of_images = len(cameleon_set)
    for image in cameleon_set:
        source = cv2.imread(image)
        gray_source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(gray_source, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray_target, None)
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict())
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

        best = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                best.append(m)

        sourcep = []
        targetp = []
        for p in best:
            sourcep.append(keypoints1[p.queryIdx].pt)
            targetp.append(keypoints2[p.trainIdx].pt)

        sourcep = np.array(sourcep, dtype=np.float32).reshape((-1, 1, 2))
        targetp = np.array(targetp, dtype=np.float32).reshape((-1, 1, 2))
        h, mask = cv2.findHomography(sourcep, targetp, cv2.RANSAC, 5.0)
        height, weidth = gray_target.shape
        warped_image = cv2.warpPerspective(source, h, (weidth, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        averageim += warped_image / num_of_images

    averageim = np.array(np.round(averageim), dtype=np.uint8)
    averageim = cv2.resize(averageim, (0, 0), fx=0.7, fy=0.7)
    cv2.imshow('Average image', averageim)
    cv2.waitKey(0)




