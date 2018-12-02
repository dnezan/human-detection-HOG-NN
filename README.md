# Human Detection using HOG and NN
    
This is an open source implementation of Human Detection using Histogram Oriented Gradient and a two layer perceptron neural network for detecting human in 2D color
images. The project consists of four steps:  
• Converting color image to black and white    
• Gradient operation (Prewitt's Operator)  
• Compute HOG features  
• Backpropogation using a two-layer perceptron  
  
There are 20 training images (10 positive and 10 negative) and 10 test images (5 positive and 5 negative) in .bmp format.  All images are of size 160 (Height) X 96 (Width). Using the parameters specified in the project description, you should have 20 X 12 cells and 19 X 11 blocks. The size of your final HOG descriptor should be 7,524 X 1.  
  
## How to compile and run the program
The only libraries used in this program is PIL and scipy in order to read and write the image, numpy in order to save the 0-255 value of each pixel location, and math to compute the square root. No other libraries or in built functions are required for any operation including convolution. 

## Functions
### Converting Color Image to Black and White  
Converted the color image into a greyscale image using the formula I =(0.299𝑅 + 0.587𝐺 + 0.114𝐵) where R, G and B are the pixel values from the red, green
and blue channels of the color image.

