import cv2
import numpy as np

# img = cv2.imread('assets/bola.jpg', 1)
# img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

cap = cv2.VideoCapture(0)

while True : 
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    image = np.zeros(frame.shape, np.uint8)
    smaller_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    image[:height//2, :width//2] = smaller_frame
    image[height//2:, :width//2] = smaller_frame
    image[:height//2, width//2:] = smaller_frame
    image[height//2:, width//2:] = smaller_frame

    cv2.imshow('frame', image)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()