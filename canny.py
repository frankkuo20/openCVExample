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


cv2.namedWindow('canny')

cv2.createTrackbar('minCanny', 'canny', 5, 500, nothing)
cv2.createTrackbar('maxCanny', 'canny', 5, 500, nothing)



while True:

    _, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.imread('data/temp.png', cv2.IMREAD_GRAYSCALE)
    image2 = image
    cv2.imshow('image', image)

    image = cv2.GaussianBlur(image, (3, 3), 0)

    image = cv2.absdiff(image, image2)
    cv2.imshow('diff', image)

    minCanny = cv2.getTrackbarPos('minCanny', 'canny')
    maxCanny = cv2.getTrackbarPos('maxCanny', 'canny')
    # image = auto_canny(image)
    image = cv2.Canny(image, minCanny, maxCanny)
    cv2.imshow('canny', image)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

