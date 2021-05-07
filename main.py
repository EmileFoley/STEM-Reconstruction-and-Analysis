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
#filenames = os.listdir(pathopen)
os.system('cls') # clear console

print(f"File names:\n{filenames}")
print(f"Length of number of files: {len(filenames)}")

imagenumber = 12

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

if width > 1024:
	if width == 2048:
		n = 4
	if width == 4096:
		n = 16
	if width == 8192:
		n = 64

else:
	n = 1

np_img_being_edited = np.array(orig_image)
cv2.imwrite(f"{filepathsave}/Original.png", orig_image)
if n > 1:
	tiles = image_slicer.slice(f"{filepathsave}/Original.png", n, save=False)
	print("=========================")
	print(tiles)
	print("=========================")
	print(tiles[0].image)
	print("=========================")
	if os.path.isdir(f"{filepathsave}/tiles") == False:
		os.mkdir(f"{filepathsave}/tiles")
	image_slicer.save_tiles(tiles, directory=f"{filepathsave}/tiles", prefix='tile', format='png')

	for x in range(n):
			np_img_being_edited = np.array(tiles[x].image)
			width, height = np_img_being_edited.shape
			print(f"Size of {tiles[x]}: {width} x {height} pixels.")
			#np_img_being_edited = pre_processing.apply(np_img_being_edited, width, height) # apply pre-processing algorithms
			np_img_being_edited, listofcoordstoinfill, numberofzeropixel = algorithms.coordslist(np_img_being_edited, width, height,filepathsave) # list the coords of zero values
			if numberofzeropixel > 4:
				np_img_being_edited = algorithms.GDI1(np_img_being_edited, listofcoordstoinfill, width, height,filepathsave) # Cubic interpolation method

				#np_img_being_edited_CLEAN = algorithms.CLEAN(np_img_being_edited, listofcoordstoinfill, width, height,filepathsave) # Clean deconvolution method WIP
				tiles[x].image = Image.fromarray(np_img_being_edited)
			else:
				print(f"Less than 4 pixels of value 0 found to be infilled. Skipping this Tile!")


	print("All tiles have been processed! Time to stitch the tiles back together")
	joined = image_slicer.join(tiles)
	print(f"Joined: {joined}")	
	np_img_being_edited = np.array(joined)
	np_img_being_edited = post_processing.apply(np_img_being_edited, width, height,filepathsave) # apply infill algorithms

	np_img_being_edited = cv2.cvtColor(np_img_being_edited, cv2.COLOR_RGBA2GRAY)
	np_img_being_edited_found = algorithms.find3(np_img_being_edited, listofcoordstoinfill, width, height,filepathsave)

else: 
	#np_img_being_edited = pre_processing.apply(np_img_being_edited, width, height) # apply pre-processing algorithms
	np_img_being_edited, listofcoordstoinfill, numberofzeropixel = algorithms.coordslist(np_img_being_edited, width, height,filepathsave) # list the coords of zero values



	if numberofzeropixel != 0:
		np_img_being_edited = algorithms.GDI1(np_img_being_edited, listofcoordstoinfill, width, height,filepathsave) # Cubic interpolation method
		cv2.imwrite(f"{filepathsave}/Algorithm_processed.png", np_img_being_edited)
		
		#np_img_being_edited_CLEAN = algorithms.CLEAN(np_img_being_edited, listofcoordstoinfill, width, height,filepathsave) # Clean deconvolution method WIP


	else:
		print(f"No pixels of value 0 found to be infilled.")
	np_img_being_edited = post_processing.apply(np_img_being_edited, width, height,filepathsave) # apply infill algorithms



	np_img_being_edited_CLEAN = algorithms.find3(np_img_being_edited, listofcoordstoinfill, width, height,filepathsave)

print(f"Main.py Finished for image number {imagenumber+1}/{len(filenames)} !")

