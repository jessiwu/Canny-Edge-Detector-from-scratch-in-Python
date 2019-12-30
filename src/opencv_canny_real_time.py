import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Canny Edge detection
    gray_edges = cv2.Canny(gray, 100, 150)

    gray_edges_200 = cv2.Canny(gray, 100, 200)

    gray_edges_300 = cv2.Canny(gray, 100, 300)

    bgr_edges = cv2.cvtColor(gray_edges, cv2.COLOR_GRAY2BGR)


    cv2.imshow('frame', frame)
    cv2.imshow('edges', gray_edges)
    cv2.imshow('edges 200', gray_edges_200)
    cv2.imshow('edges 300', gray_edges_300)

    # break if the 'q' is pressed
    tmp = cv2.waitKey(1) & 0xFF
    if tmp == ord('q'):
        break
    elif tmp == ord('b'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
