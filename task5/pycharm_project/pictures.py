import scipy.misc as img
from skimage.color import rgb2gray
image = img.imread('img.png')
image = rgb2gray(image)
print type(image)
print image.shape
image = image.reshape(-1, 28 * 28)
print image.shape
image = image.reshape(28, 28)
print image.shape