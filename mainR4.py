import numpy as np
import cv2
import cv2.aruco as aruco
import os
import utils
import math
import ContourUtils
import CalibrationWithUncertainty
from scipy.spatial import distance as dist
import gui
import matplotlib.pyplot as plt
import pickle
import glob

##################################Draw Test##########################
startPointList = []
endPointList = []
distanceList = []

drawing = False
x = 0
y = 0
ix = 0
iy = 0
x_Start = 0
y_Start = 0
# Adding Function Attached To Mouse Callback
def draw(event,x,y,flags,params):
    global ix,iy,drawing,x_Start,y_Start          #explicitly tell that they need to be global variables
    # Left Mouse Button Down Pressed
    if(event==cv2.EVENT_LBUTTONDOWN):   #save start pos of line
        drawing = True
        x_Start = x
        y_Start = y
        #print("Assigned Start Values", y_Start*scaleFactor, x_Start*scaleFactor)
    if(event==cv2.EVENT_MOUSEMOVE):
        if(drawing==True):      #update new pos
            ix = x
            iy = y
            # For Drawing Rectangle
    if(event==cv2.EVENT_LBUTTONUP):     #save End pos and draw
        #print("Assigned End Values", ix*scaleFactor, iy*scaleFactor)
        cv2.circle(undist,(int(ix*scaleFactor), int(iy*scaleFactor)),int(circleRadius*scaleFactor), (255, 255, 0), int(circleThikness*scaleFactor))     #because Lines are drawn on downscaled image always adjust for that!!
        cv2.circle(undist, (int(x_Start * scaleFactor), int(y_Start * scaleFactor)), int(circleRadius*scaleFactor), (255, 255, 0), int(circleThikness * scaleFactor))
        cv2.line(undist, pt1=(int(ix*scaleFactor), int(iy*scaleFactor)), pt2=(int(x_Start*scaleFactor), int(y_Start*scaleFactor)), color=(0, 255, 0), thickness=int(4*scaleFactor))
        startPointList.append((x_Start,y_Start))
        endPointList.append((ix,iy))
        d = dist.euclidean((int(ix*scaleFactor), int(iy*scaleFactor)), (int(x_Start*scaleFactor), int(y_Start*scaleFactor)))    #calculate the distance in pixels
        distanceList.append(d)
        midX = (int(x_Start*scaleFactor) + int(ix*scaleFactor))/2
        midY = (int(y_Start*scaleFactor)+int(iy*scaleFactor))/2         #find midpoint to writhe the number
        distance = d / pixelsPerMetricUndist    #convert distance to mm
        cv2.putText(undist, "{:.3f}".format(round(distance, 3)), (int(midX), int(midY)), cv2.FONT_HERSHEY_SIMPLEX,
                    textSize * scaleFactor, (0, 0, 255),thickness=int(textThikness*scaleFactor))    #write the distance on image
        drawing = False

###########################################################



textThikness = 1  #1
textSize = 1     #0.7
circleRadius = 4   #0.8
circleThikness = -1  #-1 fore filled circle
lineThikness = 1   #1

rows = 17            #17   6
columns = 28         #28    9

# termination criteria for Subpixel Optimization
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
squareSize = 30#mm      Beinflusst aber in keiner weise die Matrix

#print(objp.shape)
#print(objp[:,:2].shape)
#print(objp[:,0])
#print("2")
#print(objp[:,1])

ArucoSize = 31.4       #     50.5 #in mm    53mm marker alt !!!!!

#Pathname for Test images
pathName = "C:\\Users\\Lars\\Desktop\\MessBilder\\*.TIF"
#pathName = "C:\\Users\\gudjons\\Desktop\\MessBilder\\*.TIF"

savingDirectory = "C:\\Users\\Lars\\Desktop\\MessBilder\\Gemessene"

saveImages = False
undistiortTestAfterCalib = False
saveParametersPickle = False
loadSavedParameters = True
webcam = False


cap = cv2.VideoCapture(0)

cap.set(2,1920)
cap.set(3,1080)


#OpenCV Window GUI###############################
mainImage = cv2.imread("Recources/Main Frame.PNG")
root_wind = "Object measurement"
cv2.namedWindow(root_wind)
cv2.setMouseCallback(root_wind,draw)
cv2.imshow(root_wind,mainImage)


#################################Program Starting Screen#####################################
keyEvent = cv2.waitKey(0) #next imageqq
if keyEvent == ord('1'):            #calibrate and save
    saveParametersPickle = True
    loadSavedParameters = False
elif keyEvent == ord('2'):          #just calibrate
    saveParametersPickle = False
    loadSavedParameters = False
elif keyEvent == ord('3'):      #masure
    saveParametersPickle = False
    loadSavedParameters = True
elif keyEvent == ord('q'):
    exit()
else:
    cv2.waitKey(1)



def empty(a):
    pass
slider = "Edge Detection Settings"
filters = "General Settings"
cv2.namedWindow(filters)
cv2.namedWindow(slider)
cv2.resizeWindow("General Settings",400,100)
cv2.resizeWindow("Edge Detection Settings", 640, 240)
cv2.createTrackbar("Edge Thresh Low","Edge Detection Settings", 120, 255, empty)
cv2.createTrackbar("Edge Thresh High","Edge Detection Settings", 160, 255, empty)
cv2.createTrackbar("Gaussian's","Edge Detection Settings", 1, 20, empty)
cv2.createTrackbar("Dilations","Edge Detection Settings", 1, 10, empty)
cv2.createTrackbar("Erosions","Edge Detection Settings", 1, 10, empty)
cv2.createTrackbar("minArea","Edge Detection Settings", 800, 500000, empty)
cv2.createTrackbar("Epsilon","Edge Detection Settings", 5, 40, empty)
cv2.createTrackbar("Show Filters","General Settings", 1, 1, empty)
cv2.createTrackbar("Automatic","General Settings",0,1,empty)
cv2.createTrackbar("TextSize","General Settings",100,400,empty)


######################################################################


runs = 1
if not loadSavedParameters:
    meanMTX, meanDIST, uncertaintyMTX, uncertaintyDIST = CalibrationWithUncertainty.calibrateCamera(cap=cap, rows=rows, columns=columns, squareSize=squareSize, runs=runs,
                                                                                                    saveImages=False, webcam=webcam)
if saveParametersPickle:
    pickle_out_MTX = open("PickleFiles/mtx.pickle","wb")
    pickle.dump(meanMTX,pickle_out_MTX)
    pickle_out_MTX.close()
    pickle_out_DIST = open("PickleFiles/dist.pickle","wb")
    pickle.dump(meanDIST,pickle_out_DIST)
    pickle_out_DIST.close()
    pickle_out_MTX_Un = open("PickleFiles/uncertaintyMtx.pickle", "wb")
    pickle.dump(uncertaintyMTX, pickle_out_MTX_Un)
    pickle_out_MTX_Un.close()
    pickle_out_DIST_Un = open("PickleFiles/uncertaintyDist.pickle", "wb")
    pickle.dump(uncertaintyDIST, pickle_out_DIST_Un)
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

automaticMode = True
putText = True
upscale = False         #Does not work together with manual masuremts !!!!!!!
pixelsPerMetric = 1
pixelsPerMetricUndist = 1


############################################Dummy Save Test images from webcam##################################
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
    dsize = (1920, 1080)
    cv2.imshow("Test", cv2.resize(frame, dsize))
    cv2.waitKey(200)
cv2.destroyWindow("Test")

cv2.waitKey(200)

savedImageCounter = 0

for img in images:
    original = img              #Alwasy use original after one loop
    undist = utils.undistortFunction(original, meanMTX, meanDIST)  # Undistort Image
    corners, ids, rejectedImgPoints = aruco.detectMarkers(original, aruco_dict)  # Detects AruCo Marker in Image
    cornersUndist, idsUndist, rejectedImgPointsUndist = aruco.detectMarkers(undist,aruco_dict)  # Detects AruCo Marker in Image

    corners = np.array(corners)
    reorderd = ContourUtils.reorder(corners)  # Reorders Corners TL,TR,BL,BR
    cornersUndist = np.array(cornersUndist)
    reorderdUndist = ContourUtils.reorder(cornersUndist)  # Reorders Corners TL,TR,BL,BR

    if reorderd is not None and reorderdUndist is not None:
        pixelsPerMetric = utils.calculatePixelsPerMetric(original, reorderd, ArucoSize)  # Calculates Pixels/Metric and Drwas
        pixelsPerMetricUndist = utils.calculatePixelsPerMetric(undist, reorderdUndist, ArucoSize, )  # Calculates Pixels/Metric
        # cv2.putText(img, "Pixels per mm", (5, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7,(0, 0, 255))  # Draws Pixel/Lengh variable on Image'
        # cv2.putText(img, str(pixelsPerMetric),(5, 75), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255)) #Draws Pixel/Lengh variable on Image'
        cv2.putText(undist, "Pixels per mm:", (20, 200), cv2.FONT_HERSHEY_COMPLEX, 8, (0, 0, 255),
                    thickness=5)  # Draws Pixel/Lengh variable on Image'
        cv2.putText(undist, str(pixelsPerMetricUndist), (2200, 200), cv2.FONT_HERSHEY_COMPLEX, 8,
                    (0, 0, 255), thickness=5)  # Draws Pixel/Lengh variable on Image'
    else:
        cv2.putText(original, "AruCo not correctly Detected!", (500, 1000), cv2.FONT_HERSHEY_COMPLEX, 14, (0, 0, 255),
                    thickness=14)
    aruco.drawDetectedMarkers(original, corners)  # Drwas Box around Marker
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, ArucoSize, meanMTX, meanDIST)  # größße des marker in m
    # rvecZeroDist, tvecZeroDist, _ = aruco.estimatePoseSingleMarkers(corners, 0.053, mtx, distZero)  # größße des marker in m
    if rvec is not None and tvec is not None:
        aruco.drawAxis(original, meanMTX, meanDIST, rvec, tvec, 50)  # Drwas AruCo Axis
        cv2.putText(original, "%.1f cm -- %.0f deg" % ((tvec[0][0][2] / 10), (rvec[0][0][2] / math.pi * 180)),
                    (0, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 15, (255, 0, 0), thickness=textThikness)
        ################################################Select ROI###############################
        undist, shapeROI = utils.cropImage(undist)  # Crops out the ROI
        cv2.destroyWindow("ROI selector")

        scaleFactor1 = shapeROI[0] / 1000  # Scales ROI image in a way that it always fits on FHD Screen
        scaleFactor2 = shapeROI[1] / 1850
        if scaleFactor1 > scaleFactor2:
            scaleFactor = scaleFactor1
        else:
            scaleFactor = scaleFactor2
        #############################################################################
        if abs(rvec[0][0][2] / math.pi * 180) > 3:
            cv2.putText(original, "Angle should be below 3 degrees!", (0, 800), cv2.FONT_HERSHEY_SIMPLEX, 15,
                        (0, 0, 255), thickness=textThikness)
    else:
        print("No Marker found")
        cv2.imshow("Error",cv2.resize(original,(1920,1080)))
        cv2.waitKey(5000)
        break

    #if shapeROI[0] < 800 or shapeROI[1] < 800: #activate Upscale when there are few pixels
    #    upscale = True


    #################################Adjust for upscaling !!!######################
    upscaleFactor = 4
    if upscale:
        pixelsPerMetricUndist = pixelsPerMetricUndist * upscaleFactor  #When tere are more pixels after upscale Pixels/mm is higher
        undist = cv2.resize(undist,(int(undist.shape[1] * upscaleFactor), int(undist.shape[0] * upscaleFactor)),interpolation=cv2.INTER_CUBIC)
    ###############################################################################
    while True:# Loop for every Image
        imgShowCopy = original.copy()
        undistCopy = undist.copy()
        timer = cv2.getTickCount()          #FPS Counter

        cannyLow, cannyHigh, nrGauss, minArea, errosions, dialations, epsilon, showFilters, automaticMode, textSize = gui.updateTrackBar()
        #cv2.waitKey(1)
        if automaticMode:

            keyEvent = cv2.waitKey(1)
            if keyEvent == ord('d'):
                gui.resetTrackBar()

            imgContours, conts = ContourUtils.get_contours(undistCopy,shapeROI, cThr=(cannyLow, cannyHigh), gaussFilters=nrGauss, dialations=dialations, errsoions=errosions, minArea=minArea * 20, epsilon=epsilon, draw=False, showFilters=showFilters)        #gets Contours from Image
            if len(conts) != 0:                           #source, ThresCanny, min Cont Area, Resolution of Poly Approx(0.1 rough 0.01 fine)

                for obj in conts:   #for every Contour
                    cv2.polylines(undistCopy, [obj[2]], True, (0, 255, 0),int(lineThikness*scaleFactor))        #Approxes Contours with Polylines
                    #print("Number of PolyPoints",str(obj[0]))
                    #print(obj[2])
                    colorcounter = 0
                    for i in range(len(obj[2])):            #for every contour in an image
                        colorcounter += 10
                        cv2.circle(undistCopy, (int(obj[2][i][0,0]),int(obj[2][i][0,1])), int(circleRadius*scaleFactor), (255, 255, 0), int(circleThikness*scaleFactor))      #draw approx Points
                        if i == len(obj[2])-1:  #spacial Case Distance between Last and first point
                            d = dist.euclidean((obj[2][i][0, 0], obj[2][i][0, 1]), (obj[2][0][0, 0], obj[2][0][0, 1]))  #distace between points
                            (midX, midY) = ContourUtils.midpoint(obj[2][i], obj[2][0])
                        else:
                            d = dist.euclidean((obj[2][i][0,0],obj[2][i][0,1]),(obj[2][i+1][0,0],obj[2][i+1][0,1]))
                            (midX, midY) = ContourUtils.midpoint(obj[2][i], obj[2][i + 1])
                        distance = d / pixelsPerMetricUndist
                        if putText:
                            cv2.putText(undistCopy, "{:.3f}".format(round(distance,3)),(int(midX), int(midY)), cv2.FONT_HERSHEY_SIMPLEX,textSize*scaleFactor, (0, 0, 255))


        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        dsize = (1920, 1080)

        # resize image
        #cv2.setMouseCallback("image", click_and_crop)



        outputUndist = cv2.resize(undistCopy, (int(shapeROI[1]/scaleFactor),int(shapeROI[0]/scaleFactor)), interpolation=cv2.INTER_AREA)
        cv2.putText(outputUndist, str((fps)), (5, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),thickness=2)
        cv2.putText(outputUndist, "Press h for Help", (int(outputUndist.shape[1] - outputUndist.shape[1]/2), outputUndist.shape[0] - 10),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), thickness=2)
        outputNormal = cv2.resize(imgShowCopy,dsize,interpolation=cv2.INTER_AREA)
        cv2.imshow(root_wind,outputUndist)       #Main Window
        cv2.imshow("MarkerCheck", outputNormal)
        keyEvent = cv2.waitKey(1) #next imageqq
        if keyEvent == ord('x'):
            break
        elif keyEvent == ord('h'):
            print("Opening Help")
            helpimage = cv2.imread("help.PNG")
            cv2.imshow("Help",helpimage)
        elif keyEvent == ord('s'):
            print("Saved")
            utils.writeLinestoCSV(startPointList, endPointList, distanceList)
            startPointList.clear()
            endPointList.clear()
            distanceList.clear()
            utils.saveImagesToDirectory(savedImageCounter,outputUndist,savingDirectory)
            cv2.putText(outputUndist, "Saved",
                        (200,500),
                        cv2.FONT_HERSHEY_COMPLEX, textSize*scaleFactor, (0, 255, 0), thickness=4)
            cv2.imshow(root_wind, outputUndist)  # Main Window
            cv2.waitKey(200)
            savedImageCounter +=1
        elif keyEvent == ord('q'):
            exit()
            cv2.destroyAllWindows()
            # done


cv2.destroyAllWindows()

