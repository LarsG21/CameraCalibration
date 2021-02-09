import numpy as np
import cv2
import cv2.aruco as aruco
import os
import utils
import math
import ContourUtils
import glob
import matplotlib.pyplot as plt

# termination criteria for Subpixel Optimization
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001)

scale = 0.2   #Scale Factor for FindCorners in very large images

def calibrateCamera(cap,rows,columns,squareSize,objp,runs,saveImages = False, webcam = True):
    directory1 = "C:\\Users\\gudjons\\Desktop\\Corners\\Vorher"
    directory2 = "C:\\Users\\gudjons\\Desktop\\Corners\\Nacher"

    open('repErrors.txt', 'w').close()

    print(os.getcwd())
    print('Path Exists ?')

    print(os.path.exists(directory1))
    print(os.path.exists(directory2))
    if not os.path.exists(directory1) or not os.path.exists(directory2):
        saveImages = False

    allMTX = []
    allDist = []
    allRepErr = []


    for r in range(runs):
        print('Run ', str(r+1), ' of 5')
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        counter = 0
        images = []

        #reads in Calib Images
        if webcam:#Read in images from webcam
            while True:
                succsess, img = cap.read()
                cv2.putText(img, "Press x to take an image of Calicration Pattern. Take at least 10 images from different angles", (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255))

                cv2.putText(img, "Run: {:.1f}/5".format(r+1), (210, 40), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255))
                if counter >9:
                    cv2.putText(img, "Press q for next step".format(counter), (5, 40), cv2.FONT_HERSHEY_COMPLEX, 0.45,(0, 0, 255))
                else:
                    cv2.putText(img, "Captured: {:.1f}/10".format(counter), (5, 40), cv2.FONT_HERSHEY_COMPLEX, 0.45,(0, 0, 255))
                cv2.imshow("Image", img)

                if cv2.waitKey(1) & 0xff == ord('x'):
                    cv2.putText(img, "Captured", (5, 70), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0))
                    cv2.imshow("Image", img)
                    cv2.waitKey(500)
                    images.append(img)                                  #In Array ablegen
                    if saveImages:
                        utils.saveImagesToDirectory(counter,img,directory1)
                    counter += 1
                    print("Captured")
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
        else:#Files in Folder
            pathName = "CalibrationImages/Run"+str(r+1)+"/*.TIF"
            images = [cv2.imread(file) for file in glob.glob(pathName)]

        #cv2.imshow("Image", img)
        #cv2.destroyWindow("Image")

        #shows Images
        for frame in images:            #Show Images
            dsize = (1920,1080)
            cv2.imshow("Test",cv2.resize(frame, dsize))
            cv2.waitKey(20)
        cv2.destroyWindow("Test")

        #findCorners
        counter2 = 0

        for img in images:
            if not webcam:# uses a downscaled version of image to give a first guess of corners
                original =img  #keep original
                originalGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                dsize = (int(img.shape[1]*scale) , int(img.shape[0]*scale))
                img = cv2.GaussianBlur(img,(3,3),1) #Blur before rezise to avoid alising error
                img = cv2.resize(img, dsize)
            print(img.shape)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            print("Searching for corners...")
            ret, corners = cv2.findChessboardCorners(gray, (columns,rows),None) #flags=cv2.CALIB_CB_FAST_CHECK   (sorgt aber aurch für false negatives !!!)

            # If found, add object points, image points (after refining them)
            if ret == True:
                print("                        Corners Found")
                objpoints.append(objp)
                if webcam:          #subpixeloptimizer on original image
                    corners2 = cv2.cornerSubPix(gray,corners,(22,22),(-1,-1),criteria)
                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(img, (columns, rows), corners2, ret)
                    if saveImages:
                        utils.saveImagesToDirectory(counter2, img, directory2)
                    counter2 += 1
                    cv2.imshow('img', img)
                else:   #subpixeloptimizer on original image
                    corners = corners/scale     #corners must be scaled according to scale factor
                    corners2 = cv2.cornerSubPix(originalGray,corners,(11,11),(-1,-1),criteria)
                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(original, (columns, rows), corners2, ret)
                    if saveImages:
                        utils.saveImagesToDirectory(counter2, img, directory2)
                    counter2 += 1
                    dsize = (int(img.shape[1] * scale), int(img.shape[0] * scale))
                    imgShow = cv2.resize(img, dsize)
                    cv2.imshow('img', imgShow)
                imgpoints.append(corners2)


                cv2.waitKey(200)
            else:
                print("                        No Corners Found")

        message = 'Found Corners in ' +str(counter2) + ' of ' + str(len(images))+ ' images'
        print('Detect at least 10 for optimal results')
        print(message)
        dsize = (1920, 1080)
        img = cv2.resize(img,dsize)
        cv2.putText(img, message, (50, 250), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255),thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000)
        cv2.destroyWindow("img")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        print('Matrix:')
        print(mtx)
        print('Dist:')
        print(dist)
        mean_error = 0
        meanErrorZeroDist = 0
        distZero = np.array([0,0,0,0,0],dtype=float)
        #fig, ax = plt.subplots()
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, distZero)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            meanErrorZeroDist += error
        #ax.scatter(objp[:,0]*25, objp[:,1]*25)
        #print(imgpoints[0][:,0][:,0])
        #ax.scatter(imgpoints[0][:, 0], imgpoints[0][:, 1])
        #ax.scatter(imgpoints[0][:,0][:,0],imgpoints[0][:,0][:,1])
        #plt.show()
        meanErrorZeroDist = meanErrorZeroDist / len(objpoints)
        print("Mean error between Ideal Chessboard Corners and Image Corners: {}".format(meanErrorZeroDist))
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
        mean_error = mean_error / len(objpoints)
        print("Mean error between projected Objecktpoints using distortion parameters to Points in real Image: {}".format(mean_error))

        with open('repErrors.txt', 'a') as file:

            books = ["Mean Error before calib in Run {}:\n".format(r),
                     str(meanErrorZeroDist),
                     "Mean Error after calib in Run {}:\n".format(r),
                     str(mean_error),
                     ]

            file.writelines("% s\n" % data for data in books)
            file.close()
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

    with open('repErrors.txt', 'a') as file:

        books = ["MeanMTX",
                 str(meanMTX),
                 "uncertaintyMTX",
                 str(uncertantyMTX),
                 "MeanDist",
                 str(meanDIST),
                 "uncertaintyDist",
                 str(uncertantyDIST)
                 ]

        file.writelines("% s\n" % data for data in books)
        file.close()

    cv2.destroyAllWindows()
    return meanMTX,meanDIST,uncertantyMTX,uncertantyDIST
