import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True : 
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower_red = np.array([15, 239, 31])    
    # upper_red = np.array([251, 219, 207])

    # mask = cv2.inRange(hsv, lower_red, upper_red)
    # res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()