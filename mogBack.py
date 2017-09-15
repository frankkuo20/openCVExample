import cv2
import numpy as np


def nothing(x):
    pass

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()


# cv2.namedWindow('threshold')
# cv2.createTrackbar('thres', 'threshold', 245, 255, nothing)


while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    cv2.imshow('fg', fgmask)

    # dilate = cv2.dilate(fgmask, None, iterations=1)
    # cv2.imshow('dilate', dilate)

    erode = cv2.erode(fgmask, None, iterations=1)
    cv2.imshow('erode', erode)

    # thres = cv2.getTrackbarPos('thres', 'threshold')
    #
    # _, threshold = cv2.threshold(erode, thres, 255, cv2.THRESH_BINARY)
    # cv2.imshow('threshold', threshold)


    cntr_frame, contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        # if len(approx) != 4:
        #     continue

        # markColor = (255, 0, 255)
        # cv2.drawContours(frame, [approx], -1, markColor, 2)
        # if cv2.contourArea(contour) < 500:
        #     continue
        #
        # x, y, w, h = cv2.boundingRect(contour)
        # markColor = (0, 255, 0)
        # cv2.drawContours(frame, contour, -1, markColor, 2)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), markColor, 2)

    # areas = [cv2.contourArea(c) for c in contours]
    # max_index = np.argmax(areas)
    # cnt = contours[max_index]
    # markColor = (0, 127, 255)
    # cv2.drawContours(frame, contour, -1, markColor, 2)
    # x, y, w, h = cv2.boundingRect(contour)
    # cv2.rectangle(frame, (x, y), (x + w, y + h), markColor, 2)


    cv2.imshow('frame', frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
