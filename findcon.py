import cv2
import numpy as np

def nothing(x):
    pass



if __name__ == '__main__':
    origin = cv2.imread('data/temp.png')
    

    cv2.namedWindow('frame')
    
    cv2.createTrackbar('GaussianBlur', 'frame', 1, 100, nothing)
    cv2.createTrackbar('Threshold', 'frame', 10, 100, nothing)
    cv2.createTrackbar('Dilate', 'frame', 1, 5, nothing)
    
    while(True):
        
        num = cv2.getTrackbarPos('GaussianBlur', 'frame') * 2 +1
        num2 = cv2.getTrackbarPos('Threshold', 'frame')
        num3 = cv2.getTrackbarPos('Dilate', 'frame')

        frame = cv2.cvtColor(origin.copy(), cv2.COLOR_BGR2GRAY)
        frame2 = frame.copy()
        
        frame2 =  cv2.GaussianBlur(frame2, (num, num), 0)

        cv2.imshow('GaussianBlur', frame2)

        diff = cv2.absdiff(frame, frame2.copy())

        cv2.imshow('diff', diff)
        
        _, threshold = cv2.threshold(diff, num2, 255, cv2.THRESH_BINARY )
        
        cv2.imshow('threshold', threshold)

        dilate = cv2.dilate(threshold, None, iterations=num3)       
        cv2.imshow('dilate', dilate)
        
        im2, contours, hierarchy = cv2.findContours(dilate.copy(), cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_NONE)

        origin2 = origin.copy()
        for contour in contours:
            approx = cv2.approxPolyDP(contour,30,True)
            cv2.drawContours(origin2,[approx],0,(255,0,0),-1)

            # x,y,w,h = cv2.boundingRect(contour)
        
            # markColor = (0, 0, 255)
            # cv2.drawContours(origin2, contour, -1, markColor, 2)
            # cv2.rectangle(origin2,(x,y),(x+w,y+h), markColor, 2)
        
        cv2.imshow('frame', origin2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
