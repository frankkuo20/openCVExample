# This script will detect faces via your webcam.
# Tested with OpenCV3

import cv2

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('frank2.mp4')

rec = cv2.face.createLBPHFaceRecognizer()  # 3.3.0.9
# rec = cv2.face.LBPHFaceRecognizer_create() # 3.2.0.8
rec.load('trainner.yml')

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


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
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, conf = rec.predict(gray[y:y+h, x:x+w])
        if conf > 50: # is in not == 1
            id = 'unknown'
        if id == 1:
            id = 'Frank'
        cv2.putText(frame, str(id), (x, y+h), font, 4, (0, 0, 255), 5)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
