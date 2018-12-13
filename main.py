from PIL import Image
import scipy.misc
from scipy.misc import toimage, imsave
import math
import numpy
import numpy as np

#initialize the height and width of image
we = 96  #
he = 160  #
cell_we = int(we/8)
cell_he = int(he/8)

#initialize global numpy arrays used in the Canny Edge Detector
newgradientgx = np.zeros((he, we))
newgradientgy = np.zeros((he, we))
newgradientImage = np.zeros((he, we))
unsigned = np.zeros((he, we))
tan = np.zeros((he, we))
cell = np.zeros((cell_he, cell_we, 9))
block = np.zeros((cell_he, cell_we, 9))


#function to perform Prewitt operator
def prewitt(b):
    gray_img = b
    print("The values of the read image are ")
    print(gray_img)

    # Prewitt Operator
    h, w = gray_img.shape
    # define filters
    horizontal = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    vertical = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    #offset each edge by 1
    for i in range(1, h - 1):
        for j in range(1, w - 1):
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

            newgradientgx[i, j] = round(horizontalGrad)
            newgradientgy[i, j] = round(verticalGrad)

            if(newgradientgx[i,j]==0):
                tan[i,j]=90.00
            elif (newgradientgx[i, j] == 0 and newgradientgy[i, j] == 0):
                tan[i, j] = 0.00
            else:
                tan[i,j]=math.degrees(math.atan(newgradientgy[i,j]/newgradientgx[i,j]))
                if (tan[i,j]<0):
                    tan[i,j]= tan[i,j] + 360

            mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
            newgradientImage[i, j] = mag


#function to extrcat HOG features
def hog(b):
    image = b
    for i in range(1, he - 1):
        for j in range(1, we - 1):
            if (tan[i,j] >= 180 and tan[i,j] < 360):
                unsigned[i, j] = tan[i, j] - 180
            else:
                unsigned[i, j] = tan[i, j]

    #If range from -10 to 0 is required
    '''
    for i in range(1, he - 1):
        for j in range(1, we - 1):
            if (unsigned[i, j] >= 170 and unsigned[i, j] < 180):
                unsigned[i, j] = unsigned[i, j] - 180
    '''

    #Create cells
    for i in range(1, cell_he + 1):
        for j in range(1, cell_we + 1):
            x_flag=(i * 8) - 8
            y_flag = (j * 8) - 8
            for x in range(x_flag, x_flag + 8):
                for y in range(y_flag, y_flag + 8):

                    #Storing bin values
                    if (tan[x, y] >= 0 and tan[x, y] < 20):
                        ratio = ((20 - tan[x, y])/20.0)
                        cell[i - 1, j - 1, 0] = cell[i - 1, j - 1, 1] + (ratio * newgradientImage[x, y])
                        cell[i - 1, j - 1, 1] = cell[i - 1, j - 1, 1] + ((1 - ratio) * newgradientImage[x, y])

                    if (tan[x, y] >= 20 and tan[x, y] < 40):
                        ratio = ((40 - tan[x, y])/20.0)
                        cell[i - 1, j - 1, 1] = cell[i - 1, j - 1, 1] + (ratio * newgradientImage[x, y])
                        cell[i - 1, j - 1, 2] = cell[i - 1, j - 1, 1] + ((1 - ratio) * newgradientImage[x, y])

                    if (tan[x, y] >= 40 and tan[x, y] < 60):
                        ratio = ((60 - tan[x, y]) / 20.0)
                        cell[i - 1, j - 1, 2] = cell[i - 1, j - 1, 1] + (ratio * newgradientImage[x, y])
                        cell[i - 1, j - 1, 3] = cell[i - 1, j - 1, 1] + ((1 - ratio) * newgradientImage[x, y])

                    if (tan[x, y] >= 60 and tan[x, y] < 80):
                        ratio = ((80 - tan[x, y]) / 20.0)
                        cell[i - 1, j - 1, 3] = cell[i - 1, j - 1, 1] + (ratio * newgradientImage[x, y])
                        cell[i - 1, j - 1, 4] = cell[i - 1, j - 1, 1] + ((1 - ratio) * newgradientImage[x, y])

                    if (tan[x, y] >= 80 and tan[x, y] < 100):
                        ratio = ((100 - tan[x, y]) / 20.0)
                        cell[i - 1, j - 1, 4] = cell[i - 1, j - 1, 1] + (ratio * newgradientImage[x, y])
                        cell[i - 1, j - 1, 5] = cell[i - 1, j - 1, 1] + ((1 - ratio) * newgradientImage[x, y])

                    if (tan[x, y] >= 100 and tan[x, y] < 120):
                        ratio = ((120 - tan[x, y]) / 20.0)
                        cell[i - 1, j - 1, 5] = cell[i - 1, j - 1, 1] + (ratio * newgradientImage[x, y])
                        cell[i - 1, j - 1, 6] = cell[i - 1, j - 1, 1] + ((1 - ratio) * newgradientImage[x, y])

                    if (tan[x, y] >= 120 and tan[x, y] < 140):
                        ratio = ((140 - tan[x, y]) / 20.0)
                        cell[i - 1, j - 1, 6] = cell[i - 1, j - 1, 1] + (ratio * newgradientImage[x, y])
                        cell[i - 1, j - 1, 7] = cell[i - 1, j - 1, 1] + ((1 - ratio) * newgradientImage[x, y])

                    if (tan[x, y] >= 140 and tan[x, y] < 160):
                        ratio = ((160 - tan[x, y]) / 20.0)
                        cell[i - 1, j - 1, 7] = cell[i - 1, j - 1, 1] + (ratio * newgradientImage[x, y])
                        cell[i - 1, j - 1, 8] = cell[i - 1, j - 1, 1] + ((1 - ratio) * newgradientImage[x, y])

                    if (tan[x, y] >= 160 and tan[x, y] < 180):
                        ratio = ((140 - tan[x, y]) / 20.0)
                        cell[i - 1, j - 1, 8] = cell[i - 1, j - 1, 1] + (ratio * newgradientImage[x, y])
                        cell[i - 1, j - 1, 0] = cell[i - 1, j - 1, 1] + ((1 - ratio) * newgradientImage[x, y])

    #L2 Normalization
    final_feature = np.zeros(0)
    temp = np.zeros(0)
    temp2 = 0
    count=0
    for i in range(0, cell_he-1):
        for j in range(0, cell_we-1):
            for k in range(0, 9):
                temp = numpy.append(temp, cell[i,j,k])
            for k in range(0, 9):
                temp = numpy.append(temp, cell[i + 1, j, k])
            for k in range(0, 9):
                temp = numpy.append(temp, cell[i, j + 1, k])
            for k in range(0, 9):
                temp = numpy.append(temp, cell[i+1, j+1, k])
            for k in range(0,36):
                temp2 = temp2 + temp[k]*temp[k]
            normalization_factor = math.sqrt(temp2)
            temp = np.true_divide(temp, normalization_factor)

            for k in range(0, 36):
                final_feature = numpy.append(final_feature, temp[k])
            temp2 = 0
            temp = np.zeros(0)

    print(np.shape(final_feature))


    #machine_e=np.finfo(float).eps # machine epsilon
    #print(machine_e)
    numpy.savetxt('working_files/unsigned2.txt', unsigned, delimiter=',', fmt='%i')



#Driver Program
indimage = scipy.misc.imread("test_color.bmp")
print("shape is")
print(indimage.shape)
print("o")
print(cell.shape)
#Split the numpy array into RGB channels
red=indimage[:,:,0]
green=indimage[:,:,1]
blue=indimage[:,:,2]

grey = (0.299 * red) + (0.587 * green) + (0.114 * blue)

scipy.misc.imsave('working_files/test_grey.bmp', grey)

prewitt(grey)
scipy.misc.imsave('working_files/test_x_gradient.bmp', newgradientgx)
scipy.misc.imsave('working_files/test_y_gradient.bmp', newgradientgy)
scipy.misc.imsave('working_files/test_magnitude.bmp', newgradientImage)

print(newgradientImage)
numpy.savetxt('working_files/gradient_x.txt',newgradientImage, delimiter=',', fmt='%i')
numpy.savetxt('working_files/arctan.txt',tan, delimiter=',', fmt='%i')

print("diagnostics")
print(np.shape(newgradientImage))

hog(newgradientImage)

numpy.savetxt('working_files/unsigned.txt',unsigned, delimiter=',', fmt='%i')

#print(cell)
#Diagnostic Code Here
'''
toimage(grey).show()
#toimage(newgradientgx).show()
#toimage(newgradientgy).show()
#toimage(newgradientImage).show()

#numpy.savetxt('rgb_raw_values.txt',indimage, delimiter=',', fmt='%i')

#print(indimage)
#toimage(newgradientgx).show()
#numpy.savetxt('gradient255.txt',newgradientImage, delimiter=',', fmt='%i')
#
#imsave('xgradient.bmp', newgradientgx)
#imsave('ygradient.bmp', newgradientgy)
#imsave('magnitude.bmp', newgradientImage)
'''