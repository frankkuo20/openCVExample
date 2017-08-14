import numpy as np 
import cv2


# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("test.mp4")

isRecording = True

while True:

    t0 = cap.read()[1]
    t1 = cap.read()[1]

    if not isRecording:
        continue

    gray1 = cv2.cvtColor(t0, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(t1, cv2.COLOR_BGR2GRAY)

    blur1 = cv2.GaussianBlur(gray1, (7, 7), 0)
    blur2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    # cv2.imshow("pane", blur1)
    # cv2.imshow("pane2", blur2)
    
    d = cv2.absdiff(blur1, blur2)
    # cv2.imshow("diff", d)

    ret, th = cv2.threshold( d, 10, 255, cv2.THRESH_BINARY )
    # cv2.imshow("ret", th)

    dilated = cv2.dilate(th, None, iterations=1)
    # cv2.imshow("dilated", dilated)

    _, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    '''
    areas = [cv2.contourArea(c) for c in contours] 
    max_index = np.argmax(areas) 
    cnt = contours[max_index]
    '''
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
    
        markColor = (0, 255, 0)
        cv2.drawContours(t0, contour, -1, markColor, 2)
        cv2.rectangle(t0,(x,y),(x+w,y+h), markColor,2)
    cv2.imshow("final", t0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('p'):#Pause
        isRecording = False
    if cv2.waitKey(1) & 0xFF == ord('c'):#Continue
        isRecording = True


cap.release()
cv2.destroyAllWindows()

