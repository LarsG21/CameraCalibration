import cv2
import numpy as np


def midpoint(ptA, ptB):
    """
    Calculate the midpoint of A and B
    :param ptA: 2D numpy Array
    :param ptB: 2D numpy Array
    :return: 2D numpy array
    """
    return ((ptA[0, 0] + ptB[0, 0]) * 0.5, (ptA[0, 1] + ptB[0, 1]) * 0.5)


def get_contours(img, cThr=[100, 150], gaussFilters=1, dialations=6, errsoions=2, showFilters=False,
                 minArea=100, epsilon=0.01, Cornerfilter=0, draw=False, use_gpu=True):
    """
    gets Contours from an image

    :param img: input image (numpy array)
    :param cThr: thrersholds for canny edge detector (list)
    :param gaussFilters: number of gaussian smoothing filters (int)
    :param showFilters: boolean if you want to see the filters
    :param minArea: minimum area of vontours to filter out small noise
    :param epsilon: 'resolution' of polynomial approximation of the contour
    :param Cornerfilter: Only outputs contours with n corners
    :param draw: draws detected contours on img
    :param use_gpu: use gpu for image processing
    :return: image with contours on it, (length of contour, area of contour, poly approximation, boundingbox to the contour, i)
    """
    minArea = minArea / 8000
    imgContours = img

    # turns gpu image processing on
    if use_gpu:
        imgContours = cv2.UMat(img)

    imgGray = cv2.cvtColor(imgContours, cv2.COLOR_BGR2GRAY)
    for i in range(gaussFilters):
        imgGray = cv2.GaussianBlur(imgGray, (11, 11), 1)
    if showFilters: cv2.imshow("Gauss", cv2.resize(imgGray, (int(shapeROI[1] * 2), int(shapeROI[0] * 2)),
                                                   interpolation=cv2.INTER_AREA))
    imgCanny = cv2.Canny(imgGray, cThr[0], cThr[1])
    kernel = np.ones((3, 3))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=dialations)
    imgThre = cv2.erode(imgDial, kernel, iterations=errsoions)
    if showFilters: cv2.imshow('Canny', cv2.resize(imgThre, (int(shapeROI[1] * 2), int(shapeROI[0] * 2)),
                                                   interpolation=cv2.INTER_AREA))
    contours, hiearchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            # print('minAreaFilled')
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, epsilon * peri, True)
            bbox = cv2.boundingRect(approx)
            if Cornerfilter > 0:
                if len(approx) == Cornerfilter:
                    finalCountours.append([len(approx), area, approx, bbox, i])
            else:
                finalCountours.append([len(approx), area, approx, bbox, i])
    finalCountours = sorted(finalCountours, key=lambda x: x[1], reverse=True)

    if draw:
        for con in finalCountours:
            cv2.drawContours(imgContours, con[4], -1, (0, 0, 255), 3)

    if not showFilters:
        cv2.destroyWindow("Gauss")
        cv2.destroyWindow("Canny")
    return imgContours, finalCountours


def reorder(myPoints):
    """
    Reorders a list of corner points to: top left, top right, bottom left, bottom right
    :param myPoints: list of points (np array)
    :return: reordered points (np array)
    """
    if myPoints.shape == (1, 1, 4, 2):  # 4,1,2
        myPoints = myPoints.reshape(4, 1, 2)
        myPointsNew = np.zeros_like(myPoints)
        myPoints = myPoints.reshape((4, 2))
        # print("RESHAPED_MTX",myPointsNew)
        add = myPoints.sum(1)
        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[3] = myPoints[np.argmax(add)]
        diff = np.diff(myPoints, axis=1)
        myPointsNew[1] = myPoints[np.argmin(diff)]
        myPointsNew[2] = myPoints[np.argmax(diff)]
        return myPointsNew


def warpImg(img, points, w, h, pad=20):
    # print(points)
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    imgWarp = imgWarp[pad:imgWarp.shape[0] - pad, pad:imgWarp.shape[1] - pad]
    return imgWarp


def findDis(pts1, pts2):
    return ((pts2[0] - pts1[0]) ** 2 + (pts2[1] - pts1[1]) ** 2) ** 0.5


def contour_test(image, use_gpu=True):
    """
    just for testing

    :param image:
    :param use_gpu:
    :return:
    """
    # turns gpu image processing on
    if use_gpu:
        image = cv2.UMat(image)

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for i in range(1):
        img_gray = cv2.GaussianBlur(img_gray, (11, 11), 1)

    img_canny = cv2.Canny(img_gray, 100, 150)
    kernel = np.ones((3, 3))
    img_dial = cv2.dilate(img_canny, kernel, iterations=6)
    img_er= cv2.erode(img_dial, kernel, iterations=2)

    contours, hierachy = cv2.findContours(img_er, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    corner_filter = 0
    final_contours = []

    for i in contours:
        area = cv2.contourArea(i)
        if area > 100:

            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.01 * peri, True)
            bbox = cv2.boundingRect(approx)

            if use_gpu:
                # really costly, need to get umat back to cpu to obtain shape
                len_approx = len(approx.get())
            else:
                len_approx = len(approx)

            if corner_filter > 0:
                if len_approx == corner_filter:
                    final_contours.append([len_approx, area, approx, bbox, i])
            else:
                final_contours.append([len_approx, area, approx, bbox, i])
    final_contours = sorted(final_contours, key=lambda x: x[1], reverse=True)

    return final_contours


# just to test some functionality
if __name__ == "__main__":
    print("test")
    import timeit
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # test image is 8k for now
    test_image_name = "test_image.jpg"
    test_image_path = os.path.join(dir_path, test_image_name)
    test_image = cv2.imread(test_image_path)

    # testing with gpu first
    start = timeit.default_timer()

    cont = contour_test(test_image)

    stop = timeit.default_timer()
    execution_time = stop - start

    print("GPU: Program Executed in " + str(execution_time))  # It returns time in seconds

    # testing with gpu first
    start = timeit.default_timer()

    cont_cpu = contour_test(test_image, use_gpu=False)
    stop = timeit.default_timer()
    execution_time = stop - start

    print("CPU: Program Executed in " + str(execution_time))  # It returns time in seconds


