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

ArucoSize = 53 #in mm

saveImages = False
undistiortTestAfterCalib = False
saveParametersPickle = False
loadSavedParameters = True

# termination criteria for Subpixel Optimization
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
squareSize = 30#mm      Beinflusst aber in keiner weise die Matrix
objp = np.zeros((rows*columns,3), np.float32)
objp[:,:2] = np.mgrid[0:columns,0:rows].T.reshape(-1,2)*squareSize

runs = 5
if not loadSavedParameters:
    meanMTX,meanDIST,uncertantyMTX,uncertantyDIST = CalibrationWithUncertanty.calibrateCamera(cap=cap,rows=rows,columns=columns,squareSize=squareSize,objp=objp,runs=runs,
                                                                                            saveImages=False)
if saveParametersPickle:
    pickle_out_MTX = open("PickleFiles/mtx.pickle","wb")
    pickle.dump(meanMTX,pickle_out_MTX)
    pickle_out_MTX.close()
    pickle_out_DIST = open("PickleFiles/dist.pickle","wb")
    pickle.dump(meanDIST,pickle_out_DIST)
    pickle_out_DIST.close()
    pickle_out_MTX_Un = open("PickleFiles/uncertaintyMtx.pickle", "wb")
    pickle.dump(uncertantyMTX, pickle_out_MTX_Un)
    pickle_out_MTX_Un.close()
    pickle_out_DIST_Un = open("PickleFiles/uncertaintyDist.pickle", "wb")
    pickle.dump(uncertantyDIST, pickle_out_DIST_Un)
    pickle_out_DIST_Un.close()

if loadSavedParameters:
    pickle_in_MTX = open("PickleFiles/mtx.pickle","rb")
    meanMTX = pickle.load(pickle_in_MTX)
    print(meanMTX)
    pickle_in_DIST = open("PickleFiles/dist.pickle", "rb")
    meanDIST = pickle.load(pickle_in_DIST)
    print(meanDIST)

if undistiortTestAfterCalib:
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
pixelsPerMetric = 1

while True:
    timer = cv2.getTickCount()          #FPS Counter
    succsess, img = cap.read()
    undist = utils.undistortFunction(img, meanMTX, meanDIST)    #Undistort Image
    cv2.waitKey(1)
    if showConts:
        imgContours, conts = ContourUtils.getContours(undist, cThr=(160, 200), minArea=800, epsilon=0.1, draw=False, showCanny=False)        #gets Contours from Image
        if len(conts) != 0:                           #source, ThresCanny, min Cont Area, Resolution of Poly Approx(0.1 rough 0.01 fine)
            for obj in conts:   #for every Contour
                cv2.polylines(undist, [obj[2]], True, (0, 255, 0), 1)        #Approxes Contours with Polylines
                #print("Number of PolyPoints",str(obj[0]))
                #print(obj[2])
                cv2.waitKey(100)
                for i in range(len(obj[2])):
                    cv2.circle(undist, (int(obj[2][i][0,0]),int(obj[2][i][0,1])), 1, (255, 255, 0), 2)
                    d = dist.euclidean((obj[2][i],obj[2][i]),(obj[2][i+1],obj[2][i+1]))
                    if i == len(obj[2]):
                        d = dist.euclidean((obj[2][i],obj[2][i]),(obj[2][0],obj[2][0]))
                    distance = d / pixelsPerMetric
                    (midX, midY) = ContourUtils.midpoint(obj[2][i], obj[2][i+1])
                    cv2.putText(undist, "{:.1f}".format(distance),(int(midX ), int(midY)), cv2.FONT_HERSHEY_SIMPLEX,0.45, (255, 0, 0), 1)

    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict)           #Detects AruCo Marker in Image
    #print("Corners",corners)
    corners = np.array(corners)
    reorderd = ContourUtils.reorder(corners)                                            #Reorders Corners TL,TR,BL,BR
    #print("reorderd",reorderd)
    if reorderd is not None:
        #print(reorderd[3])
        #print("SHAPE",reorderd[3].shape)
        (tltrX, tltrY) = ContourUtils.midpoint(reorderd[0], reorderd[1]) #top left,top right
        (blbrX, blbrY) = ContourUtils.midpoint(reorderd[2], reorderd[3]) #bottom left, botto right
        (tlblX, tlblY) = ContourUtils.midpoint(reorderd[0], reorderd[2])
        (trbrX, trbrY) = ContourUtils.midpoint(reorderd[1], reorderd[3])
        cv2.circle(img, (int(tltrX), int(tltrY)), 1, (255, 255, 0), 2)
        cv2.circle(img, (int(blbrX), int(blbrY)), 1, (255, 255, 0), 2)
        cv2.circle(img, (int(tlblX), int(tlblY)), 1, (255, 255, 0), 2)
        cv2.circle(img, (int(trbrX), int(trbrY)), 1, (255, 255, 0), 2)
        cv2.line(img, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),              #Draws Lines in Center
                 (255, 0, 255), 2)
        cv2.line(img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)

        dY = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dX = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        pixelsPerMetric = (dX+dY)/2*(1/ArucoSize)                     #Calculates Pixels/Lenght Parameter
        dimA = dY / pixelsPerMetric
        dimB = dX / pixelsPerMetric             #Dimention of Marker
        cv2.putText(img, "{:.1f}mm".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2)
        cv2.putText(img, "{:.1f}mm".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2)
        cv2.putText(img, "Pixels per mm", (5, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7,(0, 0, 255))  # Draws Pixel/Lengh variable on Image'
        cv2.putText(img, str(pixelsPerMetric),(5, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255)) #Draws Pixel/Lengh variable on Image'

    aruco.drawDetectedMarkers(img, corners)      #Drwas Box around Marker
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, ArucoSize, meanMTX, meanDIST)  # größße des marker in m
    # rvecZeroDist, tvecZeroDist, _ = aruco.estimatePoseSingleMarkers(corners, 0.053, mtx, distZero)  # größße des marker in m
    if rvec is not None and tvec is not None:
        aruco.drawAxis(img, meanMTX, meanDIST, rvec, tvec, 0.05)     #Drwas AruCo Axis
        cv2.putText(img, "%.1f cm -- %.0f deg" % ((tvec[0][0][2] /10), (rvec[0][0][2] / math.pi * 180)), (0, 230),
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