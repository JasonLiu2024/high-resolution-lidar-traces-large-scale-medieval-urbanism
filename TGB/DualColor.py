from PIL import Image, ImageOps
import scipy
import numpy as np
def rgb_to_gray(rgb):
	""" 
        rgb: array of [R, G, B]
        turn rgb array into GRAY array"""
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	return gray

def two_colors(filename, threshold, other_color):
	'''
		filename: name of png image
		threshold: 
		other_color: that, instead of white (for black-white img)
		RETURN: binary version
		import and then binarize data based on threshold
		'''
	im = Image.open(filename)
	imarray = np.array(im)
	# print(f"shape of image array: {np.shape(imarray)}")
	imarray = rgb_to_gray(imarray)
	# apply gaussian filter
	imarray = scipy.ndimage.gaussian_filter(imarray, 0.55)
	# print(f"shape of image array, through gaussian filter: {np.shape(imarray)}")
	x, y = imarray.shape
	data = np.zeros((x, y))
	for i in range(x):
		for j in range(y):
			data[i][j] = 1 - imarray[i][j]
	# normalize
	data += np.abs(np.min(data))
	data = data / np.max(data)
	# turn into binary, by threshold
	binary = np.ones((x, y, 3), dtype=int)
	for i in range(x):
		for j in range(y):
			if data[i][j] > threshold:
				binary[i][j] = other_color
	return binary

import cv2 as cv
def Grayscale(img, background_color):
    """ FOR: grayscales image & replaces transparent part with background color
        filename
        background_color: ALL-CAPS text e.g. WHITE"""
    background = Image.new(mode="RGBA", size=img.size, color=background_color)
    # pasting image ONTO background
    background.paste(im=img, box=(0, 0), mask=img)
    background = background.convert("RGB")
    img_grayscale = ImageOps.grayscale(background)
    # plt.matshow(img_grayscale, cmap=plt.cm.gray)
    return img_grayscale

def Threshold(img_grayscale, kernel_size=(5, 5), sigma_x=0): 
    """ FOR: applies Otsu Threshold & Gaussian Blur
        kernel_size: (int, int) 
        sigma_x: number
        \nthe params are from https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html"""
    # source: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    # cv grayscale & white background alternative below: 
    # img_grayscale_cv = cv.imread(imageFile, cv.IMREAD_GRAYSCALE) 
    # print(f"applied Gaussian blur")
    img_blurred = cv.GaussianBlur(np.asarray(img_grayscale).astype('uint8'), 
                                  kernel_size, sigma_x)
    # print(f"apply Otsu threshold")
    threshold_value, img_thresholded = cv.threshold(
        img_blurred, 0, 225, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # plt.matshow(img_thresholded, cmap=plt.cm.gray)
    return threshold_value, img_thresholded

def binary(grayscale_array, threshold, color, background_color):
	'''
		grayscale_array: ONLY 2 dimensions
		threshold: RGB value [0, 255]
		color: [R, G, B]
		background_color: [R, G, B]
		RETURN: binary bitmap ARRAY, dual-color array
		'''
	y, x = grayscale_array.shape
	bitmap = np.zeros(shape=(y, x))
	dual_color = np.ones(shape=(y, x, 3)) * color
	for row in range(y):
		for col in range(x):
			if grayscale_array[row][col] > threshold:
				bitmap[row][col] = 1
				dual_color[row][col] = background_color
	return bitmap.astype('uint8'), dual_color.astype('uint8')