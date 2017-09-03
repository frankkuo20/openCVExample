import cv2
import numpy as np


def nothing(x):
    pass


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


cap = cv2.VideoCapture(0)

# cv2.namedWindow('canny')
#
# cv2.createTrackbar('minCanny', 'canny', 50, 500, nothing)
# cv2.createTrackbar('maxCanny', 'canny', 150, 500, nothing)

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # blur2 = cv2.GaussianBlur(gray, (5, 5), 0)

    # minCanny = cv2.getTrackbarPos('minCanny', 'canny')
    # maxCanny = cv2.getTrackbarPos('maxCanny', 'canny')
    # canny = cv2.Canny(blur, minCanny, maxCanny)
    # cv2.imshow('canny', canny)

    edges = auto_canny(blur)
    cv2.imshow('edges', edges)

    dilate = cv2.dilate(edges, None, iterations=0)
    cv2.imshow('dilate', dilate)

    frame2 = frame.copy()
    frame3 = frame.copy()

    cntr_frame, contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        markColor = (0, 255, 0)
        cv2.drawContours(frame2, contour, -1, markColor, 2)
        cv2.rectangle(frame2, (x, y), (x + w, y + h), markColor, 2)

    cv2.imshow('frame2', frame2)

    cntr_frame, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

        if len(approx) != 4:
            continue

        # if cv2.contourArea(contour) < 500:
        #     continue

        markColor = (0, 255, 255)
        cv2.drawContours(frame3, [approx], -1, markColor, 2)

        # print(cv2.contourArea(contour))
        # x, y, w, h = cv2.boundingRect(contour)
        #
        # markColor = (0, 255, 0)
        # cv2.drawContours(frame3, contour, -1, markColor, 2)
        # cv2.rectangle(frame3, (x, y), (x + w, y + h), markColor, 2)

    cv2.imshow('frame3', frame3)



    # diff = cv2.absdiff(blur, blur2)
    # _, threshold = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    # cv2.imshow('threshold', threshold)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()

