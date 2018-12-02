from PIL import Image
import scipy.misc
from scipy.misc import toimage, imsave
import math
import numpy
import numpy as np

#initialize the height and width of image
we = 256  #665
he = 256  #443

#initialize global numpy arrays used in the Canny Edge Detector
newgradientgx = np.zeros((he, we))
newgradientgy = np.zeros((he, we))
newgradientImage = np.zeros((he, we))

#function to perform Prewitt operator
def prewitt(b):
    gray_img = np.array(Image.open(b)).astype(np.uint8)
    print("The values of the read image are ")
    print(gray_img)

    # Prewitt Operator
    h, w = gray_img.shape
    # define filters
    horizontal = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    vertical = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    #offset each edge by 1
    for i in range(5, h - 5):
        for j in range(5, w - 5):
            horizontalGrad = (horizontal[0, 0] * gray_img[i - 1, j - 1]) + \
                             (horizontal[0, 1] * gray_img[i - 1, j]) + \
                             (horizontal[0, 2] * gray_img[i - 1, j + 1]) + \
                             (horizontal[1, 0] * gray_img[i, j - 1]) + \
                             (horizontal[1, 1] * gray_img[i, j]) + \
                             (horizontal[1, 2] * gray_img[i, j + 1]) + \
                             (horizontal[2, 0] * gray_img[i + 1, j - 1]) + \
                             (horizontal[2, 1] * gray_img[i + 1, j]) + \
                             (horizontal[2, 2] * gray_img[i + 1, j + 1])

            verticalGrad = (vertical[0, 0] * gray_img[i - 1, j - 1]) + \
                           (vertical[0, 1] * gray_img[i - 1, j]) + \
                           (vertical[0, 2] * gray_img[i - 1, j + 1]) + \
                           (vertical[1, 0] * gray_img[i, j - 1]) + \
                           (vertical[1, 1] * gray_img[i, j]) + \
                           (vertical[1, 2] * gray_img[i, j + 1]) + \
                           (vertical[2, 0] * gray_img[i + 1, j - 1]) + \
                           (vertical[2, 1] * gray_img[i + 1, j]) + \
                           (vertical[2, 2] * gray_img[i + 1, j + 1])

            newgradientgx[i, j] = horizontalGrad
            newgradientgy[i, j] = verticalGrad

            if(newgradientgx[i,j]==0):
                tan[i,j]=90.00
            else:
                tan[i,j]=math.degrees(math.atan(newgradientgy[i,j]/newgradientgx[i,j]))
                if (tan[i,j]<0):
                    tan[i,j]= tan[i,j] + 360

            mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
            newgradientImage[i - 1, j - 1] = mag


#Driver Program
indimage = scipy.misc.imread("test_color.bmp")
print(indimage.shape)

#Split the numpy array into RGB channels
red=indimage[:,:,0]
green=indimage[:,:,1]
blue=indimage[:,:,2]

grey = (0.299 * red) + (0.587 * green) + (0.114 * blue)

scipy.misc.imsave('working_files/test_grey.bmp', grey)
toimage(grey).show()

#numpy.savetxt('rgb_raw_values.txt',indimage, delimiter=',', fmt='%i')

#print(indimage)
#toimage(newgradientgx).show()
#numpy.savetxt('gradient255.txt',newgradientImage, delimiter=',', fmt='%i')
#
#imsave('xgradient.bmp', newgradientgx)
#imsave('ygradient.bmp', newgradientgy)
#imsave('magnitude.bmp', newgradientImage)
