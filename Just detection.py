import os
import os.path
from os import path
import time
import cv2
import numpy as np
import pre_processing
import algorithms
import post_processing
import matplotlib.pyplot as plt
from scipy.signal.signaltools import wiener
import image_slicer
from PIL import Image





fileext = '.tif'                                                                                
directory = os.getcwd()
pathversion = 'V12-Image-Improvement/'
pathopen = 'input_images/'
pathsave = 'processed_images/'
filenames = os.listdir(f"{directory}/{pathversion}{pathopen}")

os.system('cls') # clear console

print(f"File names:\n{filenames}")
print(f"Length of number of files: {len(filenames)}")

imagenumber = 1
filepathopen = f"{directory}/{pathversion}{pathopen}{filenames[imagenumber]}"
filepathsave = f"{directory}/{pathversion}{pathsave}{filenames[imagenumber]}"
			
if os.path.isdir(filepathsave) == False:
		os.mkdir(filepathsave)

else:
		print('Directory for image saving already exists')



# Load the original image, and get the number of rows, columns and channels
orig_image = cv2.imread(filepathopen, cv2.IMREAD_GRAYSCALE)
width, height = orig_image.shape

# Show information about the original image.
print(f"== Image Details ==")
print(filepathopen)
print(f"Original image: {filenames[imagenumber]}")
print(f"Size: {width} x {height} pixels")

np_img_being_edited = np.array(orig_image)
cv2.imwrite(f"{filepathsave}/Original.png", orig_image)

np_img_being_edited, listofcoordstoinfill, numberofzeropixel = algorithms.coordslist(np_img_being_edited, width, height,filepathsave) # list the coords of zero values

np_img_being_edited = cv2.imread(f"{filepathsave}/Post_Processed.png", cv2.IMREAD_GRAYSCALE)
np_img_being_edited_CLEAN = algorithms.find3(np_img_being_edited, listofcoordstoinfill, width, height,filepathsave)

print(f"Main.py Finished for image number {imagenumber+1}/{len(filenames)} !")

