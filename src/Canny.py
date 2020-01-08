import numpy as np
import time
import cv2
from matplotlib import pyplot as plt

class MagicCanny:
    def __init__(self, src_image, threshold1, threshold2):
        self.src_image = src_image
        self.edges = src_image
        self.threshold_minVal = threshold1
        self.threshold_maxVal = threshold2
        self.ROWS_HEIGHT = src_image.shape[0]
        self.COLS_WIDTH = src_image.shape[1]

    def printSrcImageShape(self):
        print("The size of source image is:", end="")
        print((self.src_image).shape)

    def showBeforeAndAfter(self):
        plt.suptitle('Canndy Edge Detection')
        plt.subplot(121),plt.imshow(cv2.cvtColor(self.src_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(cv2.cvtColor(self.edges, cv2.COLOR_BGR2RGB))
        plt.title('Edges of the Input Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    def noiseReduction(self, denoise_filter_options):
        if (denoise_filter_options==0):
            # Use Bilateral Filtering for noise removal while keeping edges sharp
            denoise_img = cv2.bilateralFilter(self.src_image, 5, 75, 75) # (src, dst, filter_size,  sigmaSpace, borderType)
        elif (denoise_filter_options==1):
            # Use GaussianBlur filter to blur an image
            denoise_img = cv2.GaussianBlur(self.src_image, (5,5), 0)
        else:
            # Use the median filter to blur an image .
            denoise_img = cv2.medianBlur(self.src_image, 5)

        return denoise_img

    def sobelOperation(self, denoise_img):
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


        sobel_X_img = cv2.filter2D( denoise_img, cv2.CV_32F, Gx_mask, (-1, -1))
        sobel_Y_img = cv2.filter2D( denoise_img, cv2.CV_32F, Gy_mask, (-1, -1))

        return sobel_X_img, sobel_Y_img

    def findGradientIntensityAndDirection(self, sobel_X_img, sobel_Y_img):

        # calculate the Gradient Direction (always perpendicular to the edges) of the image
        gradient_direction = np.arctan2(sobel_Y_img, sobel_X_img) * 180 / np.pi   # from radians to degree

        # calculate the Gradient Intensity
        gradient_intensity = sobel_X_img*sobel_X_img + sobel_Y_img*sobel_Y_img
        gradient_intensity = np.sqrt(gradient_intensity, dtype=np.float32)
        gradient_intensity = gradient_intensity / gradient_intensity.max() * 255  # the value range is mapped to [0, 255]

        return gradient_intensity, gradient_direction

    def NonMaximumSuppression(self, gradient_intensity, gradient_direction):

        # convert the range of angles (in degrees) from [-180, +180] to [0, 180]
        gradient_direction = np.where(gradient_direction<0, gradient_direction+180, gradient_direction)

        new_edges = np.zeros((self.ROWS_HEIGHT, self.COLS_WIDTH), dtype=np.uint8)

        new_directions = np.where(gradient_direction < 22.5, 1, gradient_direction)
        new_directions = np.where(gradient_direction >= 157.5, 1, gradient_direction)

        for i in range(1, self.ROWS_HEIGHT-1):
            for j in range(1, self.COLS_WIDTH-1):
                try:
                    #angle 0
                    if (gradient_direction[i,j] < 22.5) or (157.5 <= gradient_direction[i,j]):
                        neighbor_1 = gradient_intensity[i, j+1]
                        neighbor_2 = gradient_intensity[i, j-1]
                    #angle 45
                    elif (gradient_direction[i,j] < 67.5):
                        neighbor_1 = gradient_intensity[i-1, j-1]
                        neighbor_2 = gradient_intensity[i+1, j+1]
                    #angle 90
                    elif (gradient_direction[i,j] < 112.5):
                        neighbor_1 = gradient_intensity[i+1, j]
                        neighbor_2 = gradient_intensity[i-1, j]
                    #angle 135
                    else:
                        neighbor_1 = gradient_intensity[i+1, j-1]
                        neighbor_2 = gradient_intensity[i-1, j+1]

                    # Check if pixel(i, j) is local maximum or not
                    if (gradient_intensity[i,j] >= neighbor_1) and (gradient_intensity[i,j] >= neighbor_2):
                        new_edges[i,j] = gradient_intensity[i,j]           # if is local maximum, set to 255?
                    else:
                        new_edges[i,j] = 0                                 # if it is not local maximum, set to 0

                except IndexError as e:
                    print('IndexError!    i is: ', end='')
                    print(i, end='')
                    print('; j is: ', end='')
                    print(j)

        return new_edges

    def DoubleThresholds(self, new_edges):

        new_edges = np.where(new_edges>self.threshold_maxVal, 255, new_edges)
        new_edges = np.where(new_edges<self.threshold_minVal, 0, new_edges)

        return new_edges

    def HysteresisTracking(self, tmp_edges):

        # find the strong pixels
        strong_i, strong_j = np.where(tmp_edges==255)
        # from the strong pixel, find if there is any weak pixels within its 8-neighbor
        for idx in range(strong_i.size):
            try:
                if (tmp_edges[strong_i[idx]+1, strong_j[idx] ] != 0):     # down
                    tmp_edges[strong_i[idx]+1, strong_j[idx]] = 255

                elif(tmp_edges[strong_i[idx]-1, strong_j[idx]] != 0):      # up
                    tmp_edges[strong_i[idx]-1, strong_j[idx]] =255

                elif(tmp_edges[strong_i[idx], strong_j[idx]-1] != 0):      # left
                    tmp_edges[strong_i[idx], strong_j[idx]-1] =255

                elif(tmp_edges[strong_i[idx], strong_j[idx]+1] != 0):      # right
                    tmp_edges[strong_i[idx], strong_j[idx]+1] =255

                elif(tmp_edges[strong_i[idx]-1, strong_j[idx]+1] != 0):      # right-up
                    tmp_edges[strong_i[idx]-1, strong_j[idx]+1] =255

                elif(tmp_edges[strong_i[idx]-1, strong_j[idx]-1] != 0):      # left-up
                    tmp_edges[strong_i[idx]-1, strong_j[idx]-1] =255

                elif(tmp_edges[strong_i[idx]+1, strong_j[idx]+1] != 0):      # right-down
                    tmp_edges[strong_i[idx]+1, strong_j[idx]+1] =255

                elif(tmp_edges[strong_i[idx]+1, strong_j[idx]-1] != 0):      # left-down
                    tmp_edges[strong_i[idx]+1, strong_j[idx]-1] =255
            except IndexError as e:
                    print('IndexError!    idx is: ', end='')
                    print(idx)
        tmp_edges = np.where(tmp_edges<255, 0, tmp_edges)

        return tmp_edges

    def CannyAlgorithm(self):

        ''' Step 1. Noise Reduction '''
        denoise_img = self.noiseReduction(1)

        ''' Step 2. Finding Intensity Gradient of the Image '''
        sobel_X_img, sobel_Y_img = self.sobelOperation(denoise_img)
        gradient_intensity, gradient_direction = self.findGradientIntensityAndDirection(sobel_X_img, sobel_Y_img)

        # ''' Step 3. Non-maximum suppression '''
        new_edges = self.NonMaximumSuppression(gradient_intensity, gradient_direction)

        ''' Step 4. Double threshold '''
        tmp_edges = self.DoubleThresholds(new_edges)

        ''' Step 5. Edge Tracking by Hysteresis '''
        res = self.HysteresisTracking(tmp_edges)

        ''' Code for test the execution time '''
        # start_time = time.time()
        # elapsed_time = time.time() - start_time
        # print(elapsed_time)

        self.edges = res
        return self.edges
        self.showBeforeAndAfter()
