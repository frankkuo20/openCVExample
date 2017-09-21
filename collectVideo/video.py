import cv2
import time
import os

type = input('1. happy \n2. Anger\nEnter type: ')
num = input('Enter student num: ')

if not os.path.exists(num):
    os.mkdir(num)


filePath = '{}/{}_{}.mp4'.format(num, num, type)

if os.path.exists(filePath):
    print('Already exist')

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use the lower case
out = cv2.VideoWriter(filePath, fourcc, 20.0, (width, height))

startBool = False
font = cv2.FONT_HERSHEY_SIMPLEX

startTime = None

while True:
    # Capture frame by frame
    ret, frame = cap.read()

    if not startBool:
        showText = str(startBool)
    else:
        out.write(frame)

        if not startTime:  # start
            startTime = time.time()

        currentTime = time.time()
        caclTime = int(currentTime - startTime)
        if caclTime == 6:
            break
        showText = '{}, {}'.format(str(startBool), str(caclTime))

    cv2.putText(frame, showText, (0, height - 5), font, 2, (0, 0, 255), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('w'):
        startBool = True

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
