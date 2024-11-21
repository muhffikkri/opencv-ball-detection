import numpy as np
import cv2

cap = cv2.VideoCapture(0)
# img = cv2.resize(cv2.imread('assets/bola.jpg', 0), (0, 0), fx=0.5, fy=0.5)
template = cv2.resize(cv2.imread('assets/tws_resized.jpg', 0), (0, 0), fx=0.2, fy=0.2)
h, w = template.shape

methods = [
    cv2.TM_CCOEFF,
    cv2.TM_CCOEFF_NORMED,
    cv2.TM_CCORR,
    cv2.TM_CCORR_NORMED,
    cv2.TM_SQDIFF,
    cv2.TM_SQDIFF_NORMED
]

while True : 

    for method in methods : 
        # img2 = img.copy()
        ret, frame = cap.read()


        # result = cv2.matchTemplate(img2, template, method)
        result = cv2.matchTemplate(frame, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            location = min_loc
        else : 
            location = max_loc

        bottom_right = (location[0] + w, location[1] + h)
        # cv2.rectangle(img2, location, bottom_right, 255, 2)
        # cv2.imshow("Match", img2)

        cv2.rectangle(frame, location, bottom_right, 255, 2)
        cv2.imshow("Frame", frame)

        if cv2.waitkey(1) == ord('q') : 
            break
 
    break
        
cv2.destroyAllWindows()