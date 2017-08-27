# This script will detect faces via your webcam.
# Tested with OpenCV3

import cv2

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('frank.mp4')

id = input('Enter user id ')

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
sampleNum = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
        # minSize=(30, 30)
        # flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        sampleNum += 1
        cv2.imwrite('dataset/User.{}.{}.jpg'.format(str(id), sampleNum),
                    gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.waitKey(100)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if sampleNum > 25:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
