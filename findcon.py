import numpy as np
import cv2

def nothing(x):
    pass


im = cv2.imread('test.png')

cv2.namedWindow('frame')
cv2.createTrackbar('GaussianBlur', 'frame' , 1, 100, nothing)
cv2.createTrackbar('Threshold', 'frame' , 10, 100, nothing)
cv2.createTrackbar('Dilate', 'frame' , 1, 5, nothing)

while (True):


    num = cv2.getTrackbarPos('GaussianBlur','frame') * 2 +1
    num2 = cv2.getTrackbarPos('Threshold','frame') 
    num3 = cv2.getTrackbarPos('Dilate', 'frame')
    
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imgray2 = imgray

    imgray =  cv2.GaussianBlur(imgray, (num, num), 0)

    imgray = cv2.absdiff(imgray, imgray2)
    ret, thresh = cv2.threshold(imgray, num2, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=num3)

    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        
        x,y,w,h = cv2.boundingRect(contour)
    
        markColor = (0, 0, 255)
        cv2.drawContours(im, contour, -1, markColor, 2)
        cv2.rectangle(im,(x,y),(x+w,y+h), markColor,2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('frame', im)
