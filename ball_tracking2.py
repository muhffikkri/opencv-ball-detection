import cv2
import numpy as np

videoCapture = cv2.VideoCapture(0)

# Define the lower and upper boundaries of the "orange" and "red" ball in the HSV color space
orangeLower = np.array([5, 100, 100])
orangeUpper = np.array([15, 255, 255])
redLower1 = np.array([0, 100, 100])
redUpper1 = np.array([10, 255, 255])
redLower2 = np.array([160, 100, 100])
redUpper2 = np.array([180, 255, 255])

while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for the orange and red colors
    maskOrange = cv2.inRange(hsv, orangeLower, orangeUpper)
    maskRed1 = cv2.inRange(hsv, redLower1, redUpper1)
    maskRed2 = cv2.inRange(hsv, redLower2, redUpper2)
    maskRed = cv2.bitwise_or(maskRed1, maskRed2)

    # Combine the masks
    mask = cv2.bitwise_or(maskOrange, maskRed)

    # Perform a series of dilations and erosions to remove any small blobs left in the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # Only proceed if at least one contour was found
    if len(contours) > 0:
        # Find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Only proceed if the radius meets a minimum size
        if radius > 10:
            # Draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # Show the frame to our screen
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # If the 'q' key is pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup the camera and close any open windows
videoCapture.release()
cv2.destroyAllWindows()