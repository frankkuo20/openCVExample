import cv2 
import numpy as np

MIN_MATCH_COUNT = 30
detector = cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDTREE = 0


flannParam = dict(algorithm=FLANN_INDEX_KDTREE, tree=5)
flann = cv2.FlannBasedMatcher(flannParam, {})

trainImg = cv2.imread('data/pi3box.jpg', 0)
trainKP, trainDecs = detector.detectAndCompute(trainImg, None)


cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture('../test.mp4')


while True:
    ret, queryImgBGR = cam.read()
    queryImg = cv2.cvtColor(queryImgBGR, cv2.COLOR_BGR2GRAY)

    queryKP, queryDecs = detector.detectAndCompute(queryImg, None)
    matches = flann.knnMatch(queryDecs, trainDecs, k=2)

    goodMatch = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            goodMatch.append(m)

    if len(goodMatch) > MIN_MATCH_COUNT:
        tp = []
        qp = []
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.trainIdx].pt)

        tp, qp = np.float32((tp, qp))
        H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
        h, w = trainImg.shape

        trainBorder = np.float32([[[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]])
        queryBorder = cv2.perspectiveTransform(trainBorder, H)
        cv2.polylines(queryImgBGR, [np.int32(queryBorder)], True,
                      (0, 255, 0), 5)
    else:
        print('not')

    cv2.imshow('result', queryImgBGR)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
