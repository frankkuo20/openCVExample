import os
import cv2
import numpy as np

recognizer = cv2.face.createLBPHFaceRecognizer()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
path = 'dataset'

def getImageWithId(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    Ids = []

    for imagePath in imagePaths:
        image = cv2.imread(imagePath, 0)
        imageNp = np.array(image, 'uint8')

        Id = int(os.path.split(imagePath)[-1].split('.')[1])
        faces = detector.detectMultiScale(imageNp)
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y + h, x:x + w])
            Ids.append(Id)

    return faceSamples, Ids




faces, Ids = getImageWithId(path)
recognizer.train(faces, np.array(Ids))
recognizer.save('trainner.yml')
