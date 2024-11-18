import cv2
import numpy as np


cap = cv2.VideoCapture(0)

while True : 
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower_red = np.array([41, 209, 41])
    # upper_red = np.array([0, 255, 0])
    lower_red = np.array([70, 24, 118])
    upper_red = np.array([90, 44, 198])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', res)
    cv2.imshow('mask', mask)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()