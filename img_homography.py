import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

refDir = 'resources/ref/'
testDir = 'resources/test/'

refImg = 'statue.jpg'

with np.load('cam_calib/out/cam_calib.npz') as X:
    mtx, dist = [X[i] for i in ('mtx', 'dist')]

MIN_MATCH_COUNT = 10
query_img = cv.imread(os.path.join(refDir, refImg))
query_img_gray = cv.cvtColor(query_img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
query_keypoints, query_describtors = sift.detectAndCompute(query_img_gray, None)

for test_img_file in os.listdir(testDir):
    test_img = cv.imread(os.path.join(testDir, test_img_file))
    test_img_gray = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
    test_keypoints, test_describtors = sift.detectAndCompute(test_img_gray, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(query_describtors, test_describtors, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([query_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([test_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        h, w = query_img_gray.shape

        # draw edges of found image
        # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # dst = cv.perspectiveTransform(pts, M)
        # test_img = cv.polylines(test_img, [np.int32(dst)], True, (0, 0, 0), 3, cv.LINE_AA)

        # draw 3D coordinate system on found image
        pts2d = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst2d = cv.perspectiveTransform(pts2d, M)
        objPts = np.float32([[0, 0, 0], [0, h - 1, 0], [w - 1, h - 1, 0], [w - 1, 0, 0]]).reshape(-1, 3)
        axis = np.float32([[w, 0, 0], [0, h, 0], [0, 0, -h]]).reshape(-1, 3)

        ret, rvecs, tvecs = cv.solvePnP(objPts, dst2d, mtx, dist)
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

        origin = tuple(dst2d[0].ravel())
        test_img = cv.line(test_img, origin, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
        test_img = cv.line(test_img, origin, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        test_img = cv.line(test_img, origin, tuple(imgpts[2].ravel()), (0, 0, 255), 5)

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matches_mask = None

    draw_params = dict(matchColor=(150, 20, 150),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matches_mask,  # draw only inliers
                       flags=2)
    result_img = cv.drawMatches(query_img, query_keypoints, test_img, test_keypoints, good, None, **draw_params)
    plt.imshow(cv.cvtColor(result_img, cv.COLOR_BGR2RGB))
    plt.show()


