import numpy as np
import cv2
from matplotlib import pyplot as plt

class MagicCanny:

    def __init__(self, src_image, threshold1, threshold2):
        self.src_image = src_image
        self.edge = src_image
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def printSrcImageShape(self):
        print("The size of source image is:", end="")
        print((self.src_image).shape)

    def showBeforeAndAfter(self):
        plt.subplot(121),plt.imshow(cv2.cvtColor(self.src_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(cv2.cvtColor(self.edge, cv2.COLOR_BGR2RGB))
        plt.title('After CannyAlgorithm Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    def CannyAlgorithm(self):

        ''' Step 1. Noise Reduction '''

        # Use Bilateral Filtering for noise removal while keeping edges sharp
        denoise_img = cv2.bilateralFilter(self.src_image, 5, 75, 75) # (src, dst, filter_size,  sigmaSpace, borderType)

        # blur = cv2.GaussianBlur(self.src_image, (5,5), 0)
        # denoise_img = cv2.GaussianBlur(self.src_image, (5,5), 0)

        ''' Step 2. Finding Intensity Gradient of the Image '''

        # the gradient mask for x-direction
        Gx_mask = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
            ])

        # the gradient mask for y-direction
        Gy_mask = np.array([
            [ 1,  2,  1],
            [ 0,  0,  0],
            [-1, -2, -1]
            ])

        sobel_X_img = cv2.filter2D( denoise_img, -1, Gx_mask, (-1, -1) )
        sobel_Y_img = cv2.filter2D( denoise_img, -1, Gy_mask, (-1, -1) )
        sobel_X_img = sobel_X_img.astype(np.float32)
        sobel_Y_img = sobel_Y_img.astype(np.float32)

        # calculate the gradient direction (always perpendicular to the edges) of the image
        Gradient_direction = np.arctan2(sobel_Y_img, sobel_X_img)
        print('intensity:')

        # calculate the intensity gradient of the image
        Gradient_img = sobel_X_img*sobel_X_img + sobel_Y_img*sobel_Y_img
        Gradient_img = np.sqrt(Gradient_img, dtype=np.float32)


        # If the image is 32-bit floating-point, the pixel values are multiplied by 255. That is, the value range [0,1] is mapped to [0,255].
        Gradient_img = Gradient_img / Gradient_img.max()
        print(Gradient_img.dtype)

        # Gradient_img = Gradient_img / Gradient_img.max() * 255
        # Gradient_img = Gradient_img.astype(np.uint8)

        ''' Step 3. Non-maximum suppression '''
        self.edge = Gradient_img
        self.showBeforeAndAfter()
