import numpy as np
import cv2
import cv2.aruco as aruco
import os
import utils
import math
import ContourUtils
import CalibrationWithUncertanty
from scipy.spatial import distance as dist

import pickle
import glob


cap = cv2.VideoCapture(0)

cap.set(2,1920)
cap.set(3,1080)

rows = 6            #17
columns = 9         #28
saveImages = False

# termination criteria for Subpixel Optimization
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
squareSize = 30#mm      Beinflusst aber in keiner weise die Matrix
objp = np.zeros((rows*columns,3), np.float32)
objp[:,:2] = np.mgrid[0:columns,0:rows].T.reshape(-1,2)*squareSize

runs = 1

meanMTX,meanDIST,uncertantyMTX,uncertantyDIST = CalibrationWithUncertanty.calibrateCamera(cap=cap,rows=rows,columns=columns,squareSize=squareSize,objp=objp,runs=runs,
                                                                                          saveImages=False)
print("Take picture to undistort")
while True:
    succsess, img = cap.read()
    cv2.imshow("Calib_Chess", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    if cv2.waitKey(1) & 0xff == ord('x'):
        succsess,image = cap.read()
        cv2.imshow("Distorted",image)
        if saveImages:
            utils.saveImagesToDirectory("_distorted",image, "C:\\Users\\Lars\\Desktop\\TestBilder")
        undist = utils.undistortFunction(image,meanMTX,meanDIST)
        cv2.imshow("Undistorted",undist)
        if saveImages:
            utils.saveImagesToDirectory("_undistorted",undist, "C:\\Users\\Lars\\Desktop\\TestBilder")
        cv2.waitKey(2000)
    cv2.waitKey(1)

cv2.destroyAllWindows()
print('LiveView')

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

distZero = np.array([0, 0, 0, 0, 0], dtype=float)

showConts = True

while True:
    timer = cv2.getTickCount()
    succsess, img = cap.read()
    # cv2.imshow("DistortedLive", img)
    undist = utils.undistortFunction(img, meanMTX, meanDIST)
    cv2.waitKey(1)
    if showConts:
        imgContours, conts = ContourUtils.getContours(undist, cThr=(220, 250), draw=True)
        if len(conts) != 0:
            for obj in conts:
                cv2.polylines(imgContours, [obj[2]], True, (0, 255, 0), 2)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(undist, aruco_dict)
    print("Corners",corners)
    corners = np.array(corners)
    #print("Shape", corners.shape)
    #print("CornersNEWSHAPE", corners)
    reorderd = ContourUtils.reorder(corners)
    print("reorderd",reorderd)
    if reorderd is not None:
        print(reorderd[3])
        print("SHAPE",reorderd[3].shape)
        (tltrX, tltrY) = ContourUtils.midpoint(reorderd[0], reorderd[1]) #top left,top right
        (blbrX, blbrY) = ContourUtils.midpoint(reorderd[2], reorderd[3]) #bottom left, botto right
        (tlblX, tlblY) = ContourUtils.midpoint(reorderd[0], reorderd[2])
        (trbrX, trbrY) = ContourUtils.midpoint(reorderd[1], reorderd[3])
        cv2.circle(undist, (int(tltrX), int(tltrY)), 1, (255, 255, 0), 2)
        cv2.circle(undist, (int(blbrX), int(blbrY)), 1, (255, 255, 0), 2)
        cv2.circle(undist, (int(tlblX), int(tlblY)), 1, (255, 255, 0), 2)
        cv2.circle(undist, (int(trbrX), int(trbrY)), 1, (255, 255, 0), 2)
        cv2.line(undist, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv2.line(undist, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)

        dY = int(dist.euclidean((tltrX, tltrY), (blbrX, blbrY)))
        dX = int(dist.euclidean((tlblX, tlblY), (trbrX, trbrY)))
        cv2.putText(undist, str(dY),(5, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255))
        cv2.putText(undist, str(dX), (5, 75), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255))

    aruco.drawDetectedMarkers(undist, corners)
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.053, meanMTX, meanDIST)  # größße des marker in m
    # rvecZeroDist, tvecZeroDist, _ = aruco.estimatePoseSingleMarkers(corners, 0.053, mtx, distZero)  # größße des marker in m
    if rvec is not None and tvec is not None:
        aruco.drawAxis(undist, meanMTX, meanDIST, rvec, tvec, 0.05)
        cv2.putText(undist, "%.1f cm -- %.0f deg" % ((tvec[0][0][2] * 100), (rvec[0][0][2] / math.pi * 180)), (0, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (244, 244, 244))
        #print(rvec)
        #print(tvec)
    else:
        print("No Marker found")
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(undist, str(int(fps)), (5, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0))
    cv2.imshow("UndistortedLive", undist)
    cv2.imshow("DistortedLive", img)

cv2.destroyAllWindows()