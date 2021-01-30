import cv2
import numpy as np

def midpoint(ptA, ptB):
	return ((ptA[0,0] + ptB[0,0]) * 0.5, (ptA[0,1] + ptB[0,1]) * 0.5)


def getContours(img, cThr=[100, 100], showCanny=False, minArea=100, filter=0, draw=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3),1)
    imgBlur2 = cv2.GaussianBlur(imgBlur, (3, 3), 1)
    imgCanny = cv2.Canny(imgBlur2, cThr[0], cThr[1])
    kernel = np.ones((3, 3))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThre = cv2.erode(imgDial, kernel, iterations=2)
    if showCanny: cv2.imshow('Canny', imgCanny)
    contours, hiearchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            #print('minAreaFilled')
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.01 * peri, True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalCountours.append([len(approx), area, approx, bbox, i])
            else:
                finalCountours.append([len(approx), area, approx, bbox, i])
    finalCountours = sorted(finalCountours, key=lambda x: x[1], reverse=True)
    #print('nowdraw')
    if draw:
        for con in finalCountours:
            #print('contour draw')
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)
    return img, finalCountours

def reorder(myPoints):
    if myPoints.shape == (1,1,4,2):   #4,1,2
        myPoints = myPoints.reshape(4,1,2)
        myPointsNew = np.zeros_like(myPoints)
        myPoints = myPoints.reshape((4,2))
        print("RESHAPED_MTX",myPointsNew)
        add = myPoints.sum(1)
        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[3] = myPoints[np.argmax(add)]
        diff = np.diff(myPoints,axis=1)
        myPointsNew[1]= myPoints[np.argmin(diff)]
        myPointsNew[2] = myPoints[np.argmax(diff)]
        return myPointsNew

def warpImg (img,points,w,h,pad=20):
    # print(points)
    points =reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad,pad:imgWarp.shape[1]-pad]
    return imgWarp

def findDis(pts1,pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5