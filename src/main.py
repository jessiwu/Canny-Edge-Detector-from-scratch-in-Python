from Canny import MagicCanny
import numpy as np
import cv2

def main():

    cap = cv2.VideoCapture(0)

    while(True):
        # Modify the frame's width and height to 1080x720
        # ret = cap.set(3, 1280)
        # ret = cap.set(4, 720)

        # Check if camera opened successfully
        if (cap.isOpened()==False):
            print("Error opening video stream or file")
            exit()

        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        magicCanny = MagicCanny(gray, 20, 100)
        # magicCanny.printSrcImageShape()
        edges = magicCanny.CannyAlgorithm()

        # Display the resulting frame
        cv2.imshow('frame', frame)
        cv2.imshow('edges', edges)

        waitKey = cv2.waitKey(1) & 0xFF
        if waitKey == ord('q'):
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    exit()

if __name__ == "__main__":
    main()
