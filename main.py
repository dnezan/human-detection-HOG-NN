import glob
import scipy.misc
import math
import numpy
import numpy as np
import array as arr
from numpy import exp, array, random, dot

#initialize the height and width of image and other global variables
we = 96  #
he = 160  #
cell_we = int(we/8)
cell_he = int(he/8)
pic_number = 0
inp=[]
ff=[]
test_feature = np.array([])
assign = 0
arr = []
hog_gen=[]

#initialize global numpy arrays used in the Canny Edge Detector
newgradientgx = np.zeros((he, we))
newgradientgy = np.zeros((he, we))
newgradientImage = np.zeros((he, we))
unsigned = np.zeros((he, we))
tan = np.zeros((he, we))
cell = np.zeros((cell_he, cell_we, 9))
block = np.zeros((cell_he, cell_we, 9))
training_set_inputs = np.empty((0,7524), int)

#function to perform Prewitt operator
def prewitt(b):
    gray_img = b
    #print("The values of the read image are ")
    #print(gray_img)

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
    global hog_gen
    global training_set_inputs
    global arr
    global test_feature
    image = b
    for i in range(1, he - 1):
        for j in range(1, we - 1):
            if (tan[i,j] >= 180 and tan[i,j] < 360):
                unsigned[i, j] = tan[i, j] - 180
            else:
                unsigned[i, j] = tan[i, j]

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

    #L2 Normalization of each block
    final_feature = np.array([])
    temp = np.array([])
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
                temp2 = temp2 + (temp[k]*temp[k])
                #print(temp2)
            normalization_factor = math.sqrt(temp2)

            #Normalize each pixel
            temp = np.divide(temp, float(normalization_factor))
            final_feature = numpy.append(final_feature, temp)
            temp2 = 0
            temp = np.zeros(0)

    test_feature = final_feature
    numpy.savetxt('working_files/final_features.txt', final_feature, delimiter=',', fmt='%f')
    training_set_inputs = np.append(training_set_inputs, np.array([final_feature.flatten()]) , axis=0)


#Defining the Neural Network Architecture
class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1

class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    #Define the required activation functions
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __relu(self, x):
        return np.maximum(x, 0)

    def __relu_derivative(self, x):
        return 1. * (x > 0)

    # Training of the Neural Network
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Use the training set
            output_from_layer_1, output_from_layer_2 = self.predict(training_set_inputs)

            # Calculation of error in second layer
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculation of error in first layer
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            #Adjustment of weight
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)
            self.layer1.synaptic_weights = self.layer1.synaptic_weights + layer1_adjustment
            self.layer2.synaptic_weights = self.layer2.synaptic_weights + layer2_adjustment

    #Function to initialize the neural network
    def predict(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

#Driver Program
print("Generating HOG for Positive Dataset")
for filepath in glob.iglob('input/Train_Positive/*.bmp'):
    i = filepath[21:]
    print(i)
    indimage = scipy.misc.imread(filepath)

    #Split the numpy array into RGB channels
    red=indimage[:,:,0]
    green=indimage[:,:,1]
    blue=indimage[:,:,2]
    grey = (0.299 * red) + (0.587 * green) + (0.114 * blue)
    scipy.misc.imsave('working_files/test_grey.bmp', grey)
    prewitt(grey)
    scipy.misc.imsave('train/train_xgrad' + i, newgradientgx)
    scipy.misc.imsave('train/train_ygrad' + i, newgradientgy)
    scipy.misc.imsave('train/train_mag' + i, newgradientImage)
    hog(newgradientImage)

    numpy.savetxt('train/hog_' + i + '.csv', test_feature, delimiter=",")

    pic_number = pic_number + 1

print("Generating HOG for Negative Dataset")
for filepath in glob.iglob('input/Train_Negative/*.bmp'):
    i = filepath[21:]
    print(i)
    indimage = scipy.misc.imread(filepath)

    # Split the numpy array into RGB channels
    red = indimage[:, :, 0]
    green = indimage[:, :, 1]
    blue = indimage[:, :, 2]
    grey = (0.299 * red) + (0.587 * green) + (0.114 * blue)
    scipy.misc.imsave('working_files/test_grey.bmp', grey)
    prewitt(grey)
    scipy.misc.imsave('train/train_xgrad' + i, newgradientgx)
    scipy.misc.imsave('train/train_ygrad' + i, newgradientgy)
    scipy.misc.imsave('train/train_mag' + i, newgradientImage)
    hog(newgradientImage)
    pic_number = pic_number + 1

print(training_set_inputs)
print(np.shape(training_set_inputs))

#Seed the random number generator
random.seed(1)

# Create layer 1 (250 neurons, each with 7254 inputs)
layer1 = NeuronLayer(250, 7524)

# Create layer 2 (a single output neuron with 250 inputs)
layer2 = NeuronLayer(1, 250)

# Layers are combined
neural_network = NeuralNetwork(layer1, layer2)

# Define the output values for the training data
training_set_outputs = array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0]]).T

# Neural Network is trained using the training input and output with 10000 iterations
neural_network.train(training_set_inputs, training_set_outputs, 10000)

# Save the complete training set input data
numpy.savetxt("foo.csv", training_set_inputs, delimiter=",")

# Test the neural network with the given images
print("Testing")
for filepath in glob.iglob('input/Test_Neg/*.bmp'):
    i = filepath[15:]
    print(i)
    indimage = scipy.misc.imread(filepath)

    # Split the numpy array into RGB channels
    red = indimage[:, :, 0]
    green = indimage[:, :, 1]
    blue = indimage[:, :, 2]
    grey = (0.299 * red) + (0.587 * green) + (0.114 * blue)
    scipy.misc.imsave('working_files/test_grey.bmp', grey)
    prewitt(grey)
    scipy.misc.imsave('test/test_xgrad' + i, newgradientgx)
    scipy.misc.imsave('test/test_ygrad' + i, newgradientgy)
    scipy.misc.imsave('test/test_mag' + i, newgradientImage)
    hog(newgradientImage)
    pic_number = pic_number + 1
    hidden_state, output = neural_network.predict(test_feature)
    if (output >= 0.5):
        print("human")
    else:
        print("not human")
    print("THE OUTPUT IS")
    print(output)

for filepath in glob.iglob('input/Test_Positive/*.bmp'):
    i = filepath[20:]
    print(i)
    indimage = scipy.misc.imread(filepath)

    # Split the numpy array into RGB channels
    red = indimage[:, :, 0]
    green = indimage[:, :, 1]
    blue = indimage[:, :, 2]
    grey = (0.299 * red) + (0.587 * green) + (0.114 * blue)
    scipy.misc.imsave('working_files/test_grey.bmp', grey)
    prewitt(grey)
    scipy.misc.imsave('test/test_xgrad' + i, newgradientgx)
    scipy.misc.imsave('test/test_ygrad' + i, newgradientgy)
    scipy.misc.imsave('test/test_mag' + i, newgradientImage)
    hog(newgradientImage)
    pic_number = pic_number + 1
    hidden_state, output = neural_network.predict(test_feature)
    numpy.savetxt('test/hog_' + i + '.csv', test_feature, delimiter=",")
    #print("THE OUTPUT IS")
    print(output)
    if(output>= 0.5):
        print("human")
    else:
        print("not human")
    #print('%f' % (output))