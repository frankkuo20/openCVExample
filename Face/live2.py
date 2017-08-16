# This script will detect faces via your webcam.
# Tested with OpenCV3

import cv2

cap = cv2.VideoCapture('../test.mp4')

# Create the haar cascade
# faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceCascade = cv2.CascadeClassifier("lbpcascade_frontalface.xml")
# faceCascade = cv2.CascadeClassifier("lbpcascade_frontalface_improved.xml")
# faceCascade = cv2.CascadeClassifier("lbpcascade_profileface.xml")

areaX = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 4 )
areaY = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 4 )
areaW = areaX*3
areaH = areaY*3


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    crop_frame = frame[areaY:areaH, areaX:areaW] # [y:h, x:w]

    cv2.imshow("cropped", crop_frame)

    cv2.rectangle(frame, (areaX, areaY), (areaW, areaH), (0, 255, 0), 2)

    # Our operations on the frame come here
    gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
