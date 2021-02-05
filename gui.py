import cv2

def updateTrackBar():
    cannyLow = cv2.getTrackbarPos("Canny Threshold Low", "Edge Detection Settings")
    cannyHigh = cv2.getTrackbarPos("Canny Threshold High", "Edge Detection Settings")
    noGauss = cv2.getTrackbarPos("Number of Gauss Filters", "Edge Detection Settings")
    minArea = cv2.getTrackbarPos("Minimum Area of Contours", "Edge Detection Settings")
    epsilon = (cv2.getTrackbarPos("Epsilon (Resolution of Poly Approximation)", "Edge Detection Settings")) / 100
    showFilters = bool(cv2.getTrackbarPos("Show Filters", "Edge Detection Settings"))

    return cannyLow, cannyHigh, noGauss, minArea, epsilon, showFilters

def resetTrackBar():
    cv2.setTrackbarPos("Canny Threshold Low", "Edge Detection Settings", 120)
    cv2.setTrackbarPos("Canny Threshold High", "Edge Detection Settings", 160)
    cv2.setTrackbarPos("Number of Gauss Filters", "Edge Detection Settings", 2)
    cv2.setTrackbarPos("Minimum Area of Contours", "Edge Detection Settings", 800)
    cv2.setTrackbarPos("Epsilon (Resolution of Poly Approximation)", "Edge Detection Settings", 100)