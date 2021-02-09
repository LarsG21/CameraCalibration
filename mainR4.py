import numpy as np
import cv2
import cv2.aruco as aruco
import os
import utils
import math
import ContourUtils
import CalibrationWithUncertanty
from scipy.spatial import distance as dist
import gui
import matplotlib.pyplot as plt
import pickle
import glob


textThikness = 10  #1
textSize = 5     #0.7
circleRadius = 10   #0.8
circleThikness = 20  #1
lineThikness = 8   #1

rows = 17            #17   6
columns = 28         #28    9

# termination criteria for Subpixel Optimization
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
squareSize = 30#mm      Beinflusst aber in keiner weise die Matrix
objp = np.zeros((rows*columns,3), np.float32)
objp[:,:2] = np.mgrid[0:columns,0:rows].T.reshape(-1,2)*squareSize
#print(objp.shape)
#print(objp[:,:2].shape)
#print(objp[:,0])
#print("2")
#print(objp[:,1])

ArucoSize = 53 #in mm

#Pathname for Test images
pathName = "C:\\Users\\gudjons\\Desktop\\MessBilder\\*.TIF"   #Noch in TIF Ändern!!!


saveImages = False
undistiortTestAfterCalib = False
saveParametersPickle = False
loadSavedParameters = True
webcam = False


cap = cv2.VideoCapture(0)

cap.set(2,1920)
cap.set(3,1080)


#OpenCV Window GUI###############################
root_wind = "Object measurement"
cv2.namedWindow(root_wind)

def empty(a):
    pass
slider = "Edge Detection Settings"
cv2.namedWindow(slider)
cv2.resizeWindow("Edge Detection Settings", 640, 240)
cv2.createTrackbar("Canny Threshold Low","Edge Detection Settings", 120, 255, empty)
cv2.createTrackbar("Canny Threshold High","Edge Detection Settings", 160, 255, empty)
cv2.createTrackbar("Number of Gauss Filters","Edge Detection Settings", 2, 20, empty)
cv2.createTrackbar("Minimum Area of Contours","Edge Detection Settings", 800, 50000, empty)
cv2.createTrackbar("Epsilon (Resolution of Poly Approximation)","Edge Detection Settings", 1, 40, empty)
cv2.createTrackbar("Show Filters","Edge Detection Settings", 0, 1, empty)

######################################################################





runs = 5
if not loadSavedParameters:
    meanMTX,meanDIST,uncertantyMTX,uncertantyDIST = CalibrationWithUncertanty.calibrateCamera(cap=cap,rows=rows,columns=columns,squareSize=squareSize,objp=objp,runs=runs,
                                                                                            saveImages=False,webcam=webcam)
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
    print("Parameters Saved")

if loadSavedParameters:
    pickle_in_MTX = open("PickleFiles/mtx.pickle","rb")
    meanMTX = pickle.load(pickle_in_MTX)
    print(meanMTX)
    pickle_in_DIST = open("PickleFiles/dist.pickle", "rb")
    meanDIST = pickle.load(pickle_in_DIST)
    print(meanDIST)
    print("Parameters Loaded")

if undistiortTestAfterCalib:
    utils.undistortPicture(cap, saveImages, meanMTX, meanDIST)
    cv2.destroyAllWindows()
print('LiveView')

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

distZero = np.array([0, 0, 0, 0, 0], dtype=float)

showConts = True
putText = True
pixelsPerMetric = 1
pixelsPerMetricUndist = 1


############################################Dummy Save Test images from webcam!##################################
testing = False
if testing:
    testCounter = 0
    while True:
        print("Take Image of  Object")
        s, image = cap.read()
        cv2.imshow("Image",image)
        if cv2.waitKey(1) & 0xff == ord('x'):
            utils.saveImagesToDirectory(testCounter, image, "C:\\Users\\Lars\\Desktop\\TestBilder\\Vorher")
            cv2.putText(image, "Captured", (5, 70), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0))
            cv2.imshow("Image", image)
            cv2.waitKey(500)
            testCounter +=1
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cv2.destroyWindow("Image")

#####################################################################################################

print("loading Images...")
images = [cv2.imread(file) for file in glob.glob(pathName)]

print("showing Images...")

for frame in images:  # Show Images
    print("Test")
    dsize = (1920, 1080)
    cv2.imshow("Test", cv2.resize(frame, dsize))
    cv2.waitKey(200)
cv2.destroyWindow("Test")

cv2.waitKey(200)

for img in images:
    original = img              #Alwasy use original after one loop
    while True:
        imgShowCopy = original.copy()
        timer = cv2.getTickCount()          #FPS Counter
        undist = utils.undistortFunction(imgShowCopy, meanMTX, meanDIST)    #Undistort Image
        cv2.waitKey(1)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(imgShowCopy, aruco_dict)           #Detects AruCo Marker in Image
        cornersUndist, idsUndist, rejectedImgPointsUndist = aruco.detectMarkers(undist, aruco_dict)  # Detects AruCo Marker in Image

        if showConts:

            cannyLow, cannyHigh, noGauss, minArea, epsilon, showFilters = gui.updateTrackBar()

            keyEvent = cv2.waitKey(1)
            if keyEvent == ord('d'):
                gui.resetTrackBar()

            imgContours, conts = ContourUtils.getContours(undist, cThr=(cannyLow, cannyHigh), gaussFilters=noGauss, minArea=minArea, epsilon=epsilon, draw=False, showFilters=showFilters)        #gets Contours from Image
            if len(conts) != 0:                           #source, ThresCanny, min Cont Area, Resolution of Poly Approx(0.1 rough 0.01 fine)
                for obj in conts:   #for every Contour
                    cv2.polylines(undist, [obj[2]], True, (0, 255, 0),lineThikness)        #Approxes Contours with Polylines
                    #print("Number of PolyPoints",str(obj[0]))
                    #print(obj[2])
                    for i in range(len(obj[2])):            #for every contour in an image
                        cv2.circle(undist, (int(obj[2][i][0,0]),int(obj[2][i][0,1])), circleRadius, (255, 255, 0), circleThikness)      #draw approx Points
                        if i == len(obj[2])-1:  #spacial Case Distance between Last and first point
                            d = dist.euclidean((obj[2][i][0, 0], obj[2][i][0, 1]), (obj[2][0][0, 0], obj[2][0][0, 1]))  #distace between points
                            (midX, midY) = ContourUtils.midpoint(obj[2][i], obj[2][0])
                        else:
                            d = dist.euclidean((obj[2][i][0,0],obj[2][i][0,1]),(obj[2][i+1][0,0],obj[2][i+1][0,1]))
                            (midX, midY) = ContourUtils.midpoint(obj[2][i], obj[2][i + 1])
                        distance = d / pixelsPerMetricUndist
                        if putText:
                            cv2.putText(undist, "{:.1f}".format(distance),(int(midX ), int(midY)), cv2.FONT_HERSHEY_SIMPLEX,textSize, (0, 0, 255), textThikness)


        corners = np.array(corners)
        reorderd = ContourUtils.reorder(corners)             #Reorders Corners TL,TR,BL,BR
        cornersUndist = np.array(cornersUndist)
        reorderdUndist = ContourUtils.reorder(cornersUndist)  # Reorders Corners TL,TR,BL,BR

        if reorderd is not None and reorderdUndist is not None:
            pixelsPerMetric = utils.calculatePixelsPerMetric(imgShowCopy, reorderd, ArucoSize)  #Calculates Pixels/Metric and Drwas
            pixelsPerMetricUndist = utils.calculatePixelsPerMetric(undist, reorderdUndist, ArucoSize,)  #Calculates Pixels/Metric
            #cv2.putText(img, "Pixels per mm", (5, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7,(0, 0, 255))  # Draws Pixel/Lengh variable on Image'
            #cv2.putText(img, str(pixelsPerMetric),(5, 75), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255)) #Draws Pixel/Lengh variable on Image'
            cv2.putText(undist, "Pixels per mm", (20, 200), cv2.FONT_HERSHEY_COMPLEX, textSize*2,(0, 0, 255),thickness=textThikness)  # Draws Pixel/Lengh variable on Image'
            cv2.putText(undist, str(pixelsPerMetricUndist), (2000, 200), cv2.FONT_HERSHEY_COMPLEX, textSize*2,(0, 0, 255),thickness=textThikness)  # Draws Pixel/Lengh variable on Image'
        else:
            cv2.putText(undist, "AruCo not correctly Detected!", (20, 50), cv2.FONT_HERSHEY_COMPLEX, textSize, (0, 0, 255),thickness=textThikness)
        aruco.drawDetectedMarkers(imgShowCopy, corners)      #Drwas Box around Marker
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, ArucoSize, meanMTX, meanDIST)  # größße des marker in m
        # rvecZeroDist, tvecZeroDist, _ = aruco.estimatePoseSingleMarkers(corners, 0.053, mtx, distZero)  # größße des marker in m
        if rvec is not None and tvec is not None:
            aruco.drawAxis(imgShowCopy, meanMTX, meanDIST, rvec, tvec, 50)     #Drwas AruCo Axis
            cv2.putText(imgShowCopy, "%.1f cm -- %.0f deg" % ((tvec[0][0][2] /10), (rvec[0][0][2] / math.pi * 180)), (0, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 15, (0, 0, 0),thickness=textThikness)
            if abs(rvec[0][0][2] / math.pi * 180) > 3:
                cv2.putText(imgShowCopy,"Angle should be below 3 degrees!",(0, 260), cv2.FONT_HERSHEY_SIMPLEX, textSize, (0, 0, 255),thickness=textThikness)
            print(rvec)
            print(tvec)
        else:
            print("No Marker found")
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(undist, str((fps)), (200, undist.shape[0]-400), cv2.FONT_HERSHEY_COMPLEX, textSize, (0, 255, 0),textThikness)
        cv2.putText(undist, "Press h for Help", (undist.shape[1]-3000, undist.shape[0] - 400), cv2.FONT_HERSHEY_COMPLEX, textSize, (0, 0, 255),thickness=textThikness)
        dsize = (1920, 1080)

        # resize image
        #cv2.setMouseCallback("image", click_and_crop)

        outputUndist = cv2.resize(undist, dsize, interpolation=cv2.INTER_AREA)
        outputNormal = cv2.resize(imgShowCopy,dsize,interpolation=cv2.INTER_AREA)
        cv2.imshow(root_wind, outputUndist)       #Main Window
        cv2.imshow("MarkerCheck", outputNormal)
        keyEvent = cv2.waitKey(1) #next imageqq
        if cv2.waitKey(1) & 0xff == ord('x'):
            break

        elif keyEvent == ord('h'):
            print("Opening Help")
            helpimage = cv2.imread("help.PNG")
            cv2.imshow("Help",helpimage)
        elif keyEvent == ord('q'):
            exit()
            cv2.destroyAllWindows()
            # done


cv2.destroyAllWindows()
