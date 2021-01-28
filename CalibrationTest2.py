import numpy as np
import cv2
import cv2.aruco as aruco
import os
import utils
import math

import pickle
import glob

cap = cv2.VideoCapture(0)

cap.set(2, 1920)
cap.set(3, 1080)

# termination criteria for Subpixel Optimization
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
squareSize = 30  # mm      Beinflusst aber in keiner weise die Matrix
objp = np.zeros((17 * 28, 3), np.float32)
objp[:, :2] = np.mgrid[0:28, 0:17].T.reshape(-1, 2) * squareSize
print(objp)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

directory1 = "C:\\Users\\Lars\\Desktop\\TestBilder\\Vorher"
directory2 = "C:\\Users\\Lars\\Desktop\\TestBilder\\Nachher"

print(os.getcwd())
print('Path Exists ?')

print(os.path.exists(directory1))
print(os.path.exists(directory2))


counter = 0
images = []

# reads in Calib Images
while True:
    succsess, img = cap.read()
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xff == ord('x'):
        cv2.putText(img, "Captured", (5, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0))
        cv2.imshow("Image", img)
        cv2.waitKey(500)
        images.append(img)  # In Array ablegen
        utils.saveImagesToDirectory(counter, img, directory1)
        counter += 1
        print("Captured")
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cv2.destroyWindow("Image")

# shows Images
for frame in images:  # Show Images
    cv2.imshow("Test", frame)
    cv2.waitKey(200)
cv2.destroyWindow("Test")

# findCorners
counter2 = 0
for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (28, 17), None)
    print("FindCorners")
    # If found, add object points, image points (after refining them)
    if ret == True:
        print("        Corners Found")
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (28, 17), corners2, ret)
        utils.saveImagesToDirectory(counter2, img, directory2)
        counter2 += 1
        cv2.imshow('img', img)
        cv2.waitKey(200)
    else:
        print("         No Corners Found")
print('Found Corners in ' + str(counter2) + ' of ' + str(len(images)) + ' images')
print('Detect at least 10 for optimal results')

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print('Matrix:')
print(mtx)
print('Dist:')
print(dist.shape)
print(dist)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
    if i == 1:
        pass
        # print('Soll')
        # print(imgpoints[i])
        # print('Nach Reproduktion')
        # print(imgpoints2)

print("total error: {}".format(mean_error / len(objpoints)))

# Wait
print("Take picture to undistort")
while True:
    succsess, img = cap.read()
    cv2.imshow("Calib_Chess", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    if cv2.waitKey(1) & 0xff == ord('x'):
        succsess, image = cap.read()
        cv2.imshow("Distorted", image)
        utils.saveImagesToDirectory("_distorted", image, "C:\\Users\\Lars\\Desktop\\TestBilder")
        undist = utils.undistortFunction(image, mtx, dist)
        cv2.imshow("Undistorted", undist)
        utils.saveImagesToDirectory("_undistorted", undist, "C:\\Users\\Lars\\Desktop\\TestBilder")
        cv2.waitKey(2000)
    cv2.waitKey(1)

cv2.destroyAllWindows()
print('LiveView')

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

distZero = np.array([0, 0, 0, 0, 0], dtype=float)

while True:
    succsess, img = cap.read()
    # cv2.imshow("DistortedLive", img)
    undist = utils.undistortFunction(img, mtx, dist)
    cv2.imshow("UndistortedLive", undist)
    cv2.waitKey(1)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict)
    aruco.drawDetectedMarkers(img, corners)
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.053, mtx, dist)  # größße des marker in m
    # rvecZeroDist, tvecZeroDist, _ = aruco.estimatePoseSingleMarkers(corners, 0.053, mtx, distZero)  # größße des marker in m
    if rvec is not None and tvec is not None:
        aruco.drawAxis(img, mtx, dist, rvec, tvec, 0.05)
        cv2.putText(img, "%.1f cm -- %.0f deg" % ((tvec[0][0][2] * 100), (rvec[0][0][2] / math.pi * 180)), (0, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (244, 244, 244))
    cv2.imshow("DistortedLive", img)
    print(rvec)
    print(tvec)
cv2.destroyAllWindows()
