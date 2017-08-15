import numpy as np 
import cv2

import time 

class Camera():
    _videoPath = 'test.mp4'
    def __init__(self):
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self._videoPath)
    
    def getCamera(self):
        return self._cap    

    def release(self):
        self._cap.release()
        cv2.destroyAllWindows()


class ImageShowBase():
    _frame = None

    def getFrame(self):
        return self._frame

    def imgShow(self, isShow, title, frame):
        if isShow:
            cv2.imshow(title, frame)


class GrayImg(ImageShowBase):
    def __init__(self, frame, isShow=False):
        self._frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        super(GrayImg, self).imgShow(isShow, 'gray', self._frame)


class GaussianBlurImg(ImageShowBase):
    def __init__(self, frame, size, x, isShow):
        # third 標準差 0 = 電腦判斷
        self._frame = cv2.GaussianBlur(frame, size, x)
        super(GaussianBlurImg, self).imgShow(isShow, 'gussianBlur', self._frame)


class AbsdiffImg(ImageShowBase):
    def __init__(self, frame, frame2, isShow):
        self._frame = cv2.absdiff(frame, frame2)
        super(AbsdiffImg, self).imgShow(isShow, 'absdiff', self._frame)


class ThresholdImg(ImageShowBase):
    def __init__(self, frame, isShow):
        self._frame = cv2.threshold(frame, 10, 255, cv2.THRESH_BINARY )
        super(ThresholdImg, self).imgShow(isShow, 'threshold', self._frame[1])


class DilateImg(ImageShowBase):
    def __init__(self, frame, isShow):
        self._frame = cv2.dilate(frame, None, iterations=1)
        super(DilateImg, self).imgShow(isShow, 'dilate', self._frame)


class FindContoursImg(ImageShowBase):
    def __init__(self, frame):
        self._frame = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        

def nothing(x):
    pass

if __name__ == '__main__':
    camera = Camera().getCamera()
    start_time = time.time()
    
    while True:
        (grabbed, frame) = camera.read()
        
        gray = GrayImg(frame, False).getFrame()
        gray2 = GrayImg(frame, False).getFrame()
        if int(time.time() - start_time) == 3:
            cv2.imwrite('temp.png', gray)
        # blur = GaussianBlurImg(gray, (7, 7), 0, True).getFrame()
        blur = gray
        blur2 = GaussianBlurImg(gray2, (5, 5), 0, True).getFrame()
        
        diff = AbsdiffImg(blur, blur2, True).getFrame()
        
        _, thres = ThresholdImg(diff, False).getFrame()
        
        dilated = DilateImg(thres, True).getFrame()

        _, contours, hierarchy = FindContoursImg(dilated).getFrame()
        
        '''
        areas = [cv2.contourArea(c) for c in contours] 
        max_index = np.argmax(areas) 
        cnt = contours[max_index]
        '''
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
        
            markColor = (0, 255, 0)
            cv2.drawContours(frame, contour, -1, markColor, 2)
            cv2.rectangle(frame,(x,y),(x+w,y+h), markColor,2)

        cv2.imshow("final", frame)

        # cv2.imwrite('temp.png', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()