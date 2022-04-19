import cv as cv
import cv2
import numpy as np


if __name__ == '__main__':

    cameleon_set = ['cameleon-source_01.jpg', 'cameleon-source_02.jpg', 'cameleon-source_03.jpg',
                    'cameleon-source_04.jpg', 'cameleon-source_05.jpg', 'cameleon-source_06.jpg',
                    'cameleon-source_07.jpg']
    cameleon_target = cv2.imread('cameleon-target.jpg')

    eagle_set = ['eagle-source_01.jpg', 'eagle-source_02.jpg', 'eagle-source_03.jpg', 'eagle-source_04.jpg',
                 'eagle-source_05.jpg', 'eagle-source_06.jpg', 'eagle-source_07.jpg', 'eagle-source_08.jpg',
                 'eagle-source_09.jpg', 'eagle-source_10.jpg', 'eagle-source_11.jpg', 'eagle-source_12.jpg',
                 'eagle-source_13.jpg', 'eagle-source_14.jpg', 'eagle-source_15.jpg']
    eagle_target = cv2.imread('eagle-target.jpg')

    einstein_set = ['einstein-source_01.jpg', 'einstein-source_02.jpg', 'einstein-source_03.jpg',
                    'einstein-source_04.jpg']
    einstein_target = cv2.imread('einstein-target.jpg')

    palm_set = ['palm-source_01.jpg', 'palm-source_02.jpg', 'palm-source_03.jpg']
    palm_target = cv2.imread('palm-target.jpg')

    source = einstein_set
    target = einstein_target
    gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    tshape = target.shape
    averageim = np.zeros(tshape, float)
    num_of_images = len(source)
    for image in source:
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
    cv2.imwrite("EinsteinAVG.jpg", averageim)
    averageim = cv2.resize(averageim, (0, 0), fx=0.7, fy=0.7)
    target = cv2.resize(target, (0, 0), fx=0.7, fy=0.7)
    cv2.imshow('Target', target)
    cv2.imshow('Average image', averageim)
    cv2.waitKey(0)




