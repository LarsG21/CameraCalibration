import cv2


class Point:
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y


cap = cv2.VideoCapture(0)

cap.set(2,1280)         #Width
cap.set(3,720)         #Hight
cap.set(10,10)        #Brightness

calibrationSquareDim = 0.019    #meters
chessBoardDimentions = (6,9)

def cerateKnownBoardPositions(boardSize,squareEdgeLenght,Points):        #Array, float, ArrayofPoints
    counter = 0
    for i in range(boardSize[0]):       #hight
        for j in range(boardSize[1]):   #width
            Points[counter] = Point(j*squareEdgeLenght,i*squareEdgeLenght)
    return  Points

def getChessBoardcorners(images,foundCorners,showresults):                           #all images + bool visualize or not
    for img in images:
        pointbuffer = []
        success = cv2.findChessboardCorners(img,(9,6),pointbuffer,cv2.CALIB_CB_ADAPTIVE_THRESH)
        if success:
            foundCorners.append(pointbuffer)

        if showresults:
            cv2.drawChessboardCorners(img,(9,6),foundCorners,success)
            cv2.imshow("Corners",img)




while True:
    success, img = cap.read()
    if success:
        cv2.imshow("Video", img)                               #Video Import from Webcam
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break

cv2.waitKey()