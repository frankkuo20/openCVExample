import cv2


def nothing(x):
    pass


cap = cv2.VideoCapture(0)


cv2.namedWindow('canny')

cv2.createTrackbar('min', 'canny', 5, 500, nothing)
cv2.createTrackbar('max', 'canny', 5, 500, nothing)



while True:

    _, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.imread('data/temp.png', cv2.IMREAD_GRAYSCALE)
    image2 = image
    cv2.imshow('image', image)

    image = cv2.GaussianBlur(image, (3, 3), 0)

    image = cv2.absdiff(image, image2)
    cv2.imshow('diff', image)

    min = cv2.getTrackbarPos('min', 'canny')
    max = cv2.getTrackbarPos('max', 'canny')
    image = cv2.Canny(image, min, max)
    cv2.imshow('canny', image)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

