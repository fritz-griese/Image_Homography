import glob

import cv2 as cv
import numpy as np

chessboard_dir = 'resources/chessboard/'
output_dir = 'out/'
calib_result_img = 'IMG_20201206_175947.jpg'


def resize(img):
    height = int(img.shape[0] * 20 / 100)
    width = int(img.shape[1] * 20 / 100)
    dim = (width, height)
    return cv.resize(img, dim)


def display_corners(img, corners, ret):
    cv.drawChessboardCorners(img, (7, 6), corners, ret)
    resized = resize(img)
    cv.imshow('img', resized)
    cv.waitKey(5000)


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
print(objp)
objp = objp * 20
print(objp)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane
images = glob.glob(chessboard_dir + '*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners_sub = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # display_corners(img, corners2, ret)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv.imread(chessboard_dir + calib_result_img)
cv.imshow('normal', resize(img))
h, w = img.shape[:2]
new_cam_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
undistorted = cv.undistort(img, mtx, dist, None, new_cam_mtx)
# crop the image
x, y, w, h = roi
undistorted = undistorted[y:y + h, x:x + w]
cv.imshow('undistorted', undistorted)
cv.imwrite(output_dir + 'calibresult.png', undistorted)
cv.waitKey()

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error
print("total error: {}".format(mean_error / len(objpoints)))

np.savez(output_dir + 'cam_calib.npz', mtx=mtx, dist=dist)



