from Canny import MagicCanny
import time
import numpy as np
import cv2
INPUT_IMG_SRC_FOLDER = 'img_src/'
INPUT_IMG_NAME = 'hebe2.jpg'

def main():
    img = cv2.imread(INPUT_IMG_SRC_FOLDER + INPUT_IMG_NAME)

    # Our operations on the frame come here
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    magicCanny = MagicCanny(gray_img, 10, 20)
    magicCanny.printSrcImageShape()
    start_time = time.time()
    edges = magicCanny.CannyAlgorithm()
    elapsed_time = time.time() - start_time
    print('Time Elapsed: ', end='')
    print(elapsed_time)

    # Display the resulting frame
    # cv2.imshow('Oringinal Image', img)
    cv2.imshow('Edges of the Input Image', edges)
    edges = cv2.Canny(gray_img, 50, 140)
    cv2.imshow('CV2 Edges of the Input Image', edges)
    magicCanny.showBeforeAndAfter()

    waitKey = cv2.waitKey(0) & 0xFF
    if waitKey == ord('q'):
        exit()

if __name__ == "__main__":
    main()
