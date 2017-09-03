import cv2
import numpy as np

def nothing(x):
    pass


camera = cv2.VideoCapture(0)

if __name__ == '__main__':

    # origin = cv2.imread('data/temp.png')

    cv2.namedWindow('frame')
    cv2.namedWindow('canny')

    cv2.createTrackbar('GaussianBlur', 'frame', 1, 100, nothing)
    cv2.createTrackbar('Threshold', 'frame', 10, 100, nothing)
    cv2.createTrackbar('Dilate', 'frame', 0, 5, nothing)

    cv2.createTrackbar('minCanny', 'canny', 20, 500, nothing)
    cv2.createTrackbar('maxCanny', 'canny', 60, 500, nothing)
    
    while True:
        (grabbed, origin) = camera.read()

        num = cv2.getTrackbarPos('GaussianBlur', 'frame') * 2 +1
        num2 = cv2.getTrackbarPos('Threshold', 'frame')
        num3 = cv2.getTrackbarPos('Dilate', 'frame')

        minCanny = cv2.getTrackbarPos('minCanny', 'canny')
        maxCanny = cv2.getTrackbarPos('maxCanny', 'canny')

        frame = cv2.cvtColor(origin.copy(), cv2.COLOR_BGR2GRAY)
        frame2 = frame.copy()
        
        frame2 = cv2.GaussianBlur(frame2, (num, num), 0)

        # cv2.imshow('GaussianBlur', frame2)

        diff = cv2.absdiff(frame, frame2.copy())

        # cv2.imshow('diff', diff)
        diff = cv2.Canny(diff, minCanny, maxCanny)
        cv2.imshow('canny', diff)

        _, threshold = cv2.threshold(diff, num2, 255, cv2.THRESH_BINARY )
        
        cv2.imshow('threshold', threshold)

        kernel = np.ones((5, 5), np.uint8)
        threshold = cv2.erode(threshold, kernel, iterations=1)

        dilate = cv2.dilate(threshold, kernel, iterations=num3)
        cv2.imshow('dilate', dilate)
        
        im2, contours, hierarchy = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE)

        minNum = 100
        maxNum = 10000
        origin2 = origin.copy()
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 30, True)
            cv2.drawContours(origin2, [approx], 0, (255, 0, 0), -1)


            # if len(contour) < minNum or len(contour) > maxNum:
            #     continue
            #
            # x, y, w, h = cv2.boundingRect(contour)
            #
            # if w*h < 4000 or w*h > 100000:
            #     continue
            #
            # markColor = (0, 0, 255)
            # cv2.drawContours(origin2, contour, -1, markColor, 2)
            # cv2.rectangle(origin2, (x, y), (x+w, y+h), markColor, 2)
        
        cv2.imshow('frame', origin2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    camera.release()

