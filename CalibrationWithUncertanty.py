import numpy as np
import cv2
import os
import utils

import pickle
import glob

cap = cv2.VideoCapture(0)

cap.set(2,1920)
cap.set(3,1080)

rows = 6
columns = 9
saveImages = False

# termination criteria for Subpixel Optimization
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
squareSize = 30#mm      Beinflusst aber in keiner weise die Matrix
objp = np.zeros((rows*columns,3), np.float32)
objp[:,:2] = np.mgrid[0:columns,0:rows].T.reshape(-1,2)*squareSize

# Arrays to store object points and image points from all the images.


directory1 = "C:\\Users\\Lars\\Desktop\\TestBilder\\Vorher"
directory2 = "C:\\Users\\Lars\\Desktop\\TestBilder\\Nachher"


print(os.getcwd())
print('Path Exists ?')

print(os.path.exists(directory1))
print(os.path.exists(directory2))

allMTX = []
allDist = []
allRepErr = []
runs = 5

for i in range(runs):
    print('Run ', str(i+1), ' of 5')
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    counter = 0
    images = []

    #reads in Calib Images
    while True:
        succsess, img = cap.read()
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xff == ord('x'):
            cv2.putText(img, "Captured", (5, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0))
            cv2.imshow("Image", img)
            cv2.waitKey(500)
            images.append(img)                                  #In Array ablegen
            if saveImages:
                utils.saveImagesToDirectory(counter,img,directory1)
            counter += 1
            print("Captured")
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cv2.destroyWindow("Image")

    #shows Images
    for frame in images:            #Show Images
        cv2.imshow("Test",frame)
        cv2.waitKey(100)
    cv2.destroyWindow("Test")

    #findCorners
    counter2 = 0
    for img in images:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (columns,rows),None)
        print("FindCorners")
        # If found, add object points, image points (after refining them)
        if ret == True:
            print("        Corners Found")
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (columns,rows), corners2,ret)
            if saveImages:
                utils.saveImagesToDirectory(counter2,img,directory2)
            counter2 += 1
            cv2.imshow('img',img)
            cv2.waitKey(200)
        else:
            print("         No Corners Found")
    print('Found Corners in ' +str(counter2) + ' of ' + str(len(images))+ ' images')
    print('Detect at least 10 for optimal results')
    cv2.destroyWindow("img")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    print('Matrix:')
    print(mtx)
    print('Dist:')
    print(dist)
    mean_error = 0
    meanErrorZeroDist = 0
    distZero = np.array([0,0,0,0,0],dtype=float)
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, distZero)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        meanErrorZeroDist += error
        if i == 1:
            pass
            # print('Soll')
            # print(imgpoints[i])
            # print('Nach Reproduktion')
            # print(imgpoints2)
    print("Mean error between Ideal Chessboard Corners and Image Corners: {}".format(meanErrorZeroDist / len(objpoints)))
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
    print("Mean error between projected Objecktpoints using distortion parameters to Points in real Image: {}".format(mean_error / len(objpoints)))


    allMTX.append(mtx)
    allDist.append(dist)


MTXStack = np.stack(allMTX,axis=1)
meanMTX = np.mean(MTXStack,axis=1)
stdMTX = np.std(MTXStack,axis=1)
print(meanMTX)
print(stdMTX)

meanFx = meanMTX[0, 0]
meanFy = meanMTX[1, 1]
meanX0 = meanMTX[0, 2]
meanY0 = meanMTX[1, 2]

#print('meanFx ', meanFx)
#print('meanFy ', meanFy)
#print('meanX0 ', meanX0)
#print('meanY0 ', meanY0)

DISTStack = np.stack(allDist,axis=1)
meanDIST = np.mean(DISTStack,axis=1)
stdDist = np.std(DISTStack,axis=1)
print(meanDIST)
print(stdDist)

meanK1 = meanDIST[0,0]
meanK2 = meanDIST[0,1]
meanP1 = meanDIST[0,2]
meanP2 = meanDIST[0,3]
meanK3 = meanDIST[0,4]
#print('meanK1 ', meanDIST[0,0])
#print('meanK2 ', meanDIST[0,1])
#print('meanP1 ', meanDIST[0,2])
#print('meanP1 ', meanDIST[0,3])
#print('meanK3 ', meanDIST[0,4])

#Konfidenzintervall 95% bei 5 Samples T- Verteilung = 1,242

uncertantyMTX = 1.242*stdMTX
uncertantyDIST = 1.242*stdDist


print('Parameter inklusive Konfidenzintervalle (95%):')
print('fx: ', str(meanFx), ' +/- ', uncertantyMTX[0,0] )
print('fy: ', str(meanFy), ' +/- ', uncertantyMTX[1,1] )
print('x0: ', str(meanX0), ' +/- ', uncertantyMTX[0,2] )
print('y0: ', str(meanY0), ' +/- ', uncertantyMTX[1,2] )
print('K1: ', str(meanK1), ' +/- ', uncertantyDIST[0,0] )
print('K2: ', str(meanK2), ' +/- ', uncertantyDIST[0,1] )
print('P1: ', str(meanP1), ' +/- ', uncertantyDIST[0,2] )
print('P2: ', str(meanP2), ' +/- ', uncertantyDIST[0,3] )
print('K3: ', str(meanK3), ' +/- ', uncertantyDIST[0,4] )
#Wait
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

while True:
    succsess, img = cap.read()
    cv2.imshow("DistortedLive", img)
    undist = utils.undistortFunction(img, mtx, dist)
    cv2.imshow("UndistortedLive", undist)
    cv2.waitKey(1)
cv2.destroyAllWindows()