import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from numpy import array
from scipy.stats import entropy
from scipy import interpolate
from termcolor import colored

print("algorithms.py loaded")

def coordslist(np_pixel_map, width, height,filepathsave):
    print(f"\ncoordslist starting...")
    algorithm_np_pixel_map = np_pixel_map.copy()
    #counting the number of pixels to be infiled
    numberofzeropixel = np.count_nonzero(np_pixel_map==0)
    print(f"Number of Zero pixels to infill: {numberofzeropixel}")
    posx, posy = np.where(np_pixel_map==0)
    listofcoordstoinfill = np.column_stack((posx,posy))
    print(f"coordslist Finished!\n")
    return algorithm_np_pixel_map, listofcoordstoinfill, numberofzeropixel


def GDI1(np_pixel_map, listofcoordstoinfill, width, height ,filepathsave):
		print(f"Griddata Interpolation Starting now...")
		tarray = np_pixel_map
		tarray = tarray.astype('float')
		tarray[tarray == 0] = 'nan' # or use np.nan
		x = np.arange(0, tarray.shape[1])
		y = np.arange(0, tarray.shape[0])
		tarray[tarray==0] = np.nan # cut?
		tarray = np.ma.masked_invalid(tarray)
		xx, yy = np.meshgrid(x, y)
		x1 = xx[~tarray.mask]
		y1 = yy[~tarray.mask]
		newarr = tarray[~tarray.mask]

		print(f"Interp starting now\n")
		GD1 = interpolate.griddata((x1, y1), newarr.ravel(),(xx, yy),method='cubic')
		print(f"Interp finished\n")

		print(f"Griddata Interpolation Finished!\n")
		return GD1
		

def CLEAN(np_pixel_map, listofcoordstoinfill, width, height ,filepathsave):
		print(f"CLEAN Deconvolution Starting now...")
		tarray = np_pixel_map
		tarray = tarray.astype('float')
		tarray[tarray == 0] = 'nan' # or use np.nan
		x = np.arange(0, tarray.shape[1])
		y = np.arange(0, tarray.shape[0])
		tarray[tarray==0] = np.nan
		tarray = np.ma.masked_invalid(tarray)
		xx, yy = np.meshgrid(x, y)
		x1 = xx[~tarray.mask]
		y1 = yy[~tarray.mask]
		newarr = tarray[~tarray.mask]

		mean = 127
		std = 127
		gain = 0
		thresh = 40
		niter = 50
		guassian_noise = np.int_(np.random.normal(mean, std, np_pixel_map.shape))
		cv2.imwrite(f"{filepathsave}/gaussian_noise.png", guassian_noise)

		#GD1 = interpolate.griddata((x1, y1), newarr.ravel(),(xx, yy),method='cubic')
		#psf = cv2.imread("psf.png", cv2.IMREAD_GRAYSCALE)
		psf = np.zeros((10,10))
		psf
		guassian_noise = np.uint8(guassian_noise)
		Cleaned, res = clean.hogbom(np_pixel_map, guassian_noise, True, gain, thresh, niter)

		print(f"Cleaned image:\n{Cleaned}")
		print(f"Res:\n{res}")
		cv2.imwrite(f"{filepathsave}/Algorithm_processed_CLEAN_cleaned.png", Cleaned)
		cv2.imwrite(f"{filepathsave}/Algorithm_processed_CLEAN_res.png", res)
		print(f"CLEAN Deconvolution Finished!\n")
		return res

def find(np_pixel_map, listofcoordstoinfill, width, height ,filepathsave):
		pixel_map = np_pixel_map.copy()

		from photutils.detection import find_peaks
		print(f"finding peaks Starting now...")

		mean = np_pixel_map.mean(axis=0).mean(axis=0)
		#print(f"Mean colour: {mean}")
		peaks = find_peaks(np_pixel_map, threshold=(mean+10))

		print(peaks)
		#print(f"size: {len(peaks)}")

		peakssorted = []

		for count, peak in enumerate(peaks):
			location = (peak['x_peak'],peak['y_peak'])
			a = (6,6)
			start = np.subtract(location,a)
			end = np.add(location,a)
			cv2.rectangle(pixel_map, tuple(start), tuple(end), (0,0,255), 1)

			count2 = count + 1
			if count2 != len(peaks):
				x1 = peaks[count]['x_peak']
				y1 = peaks[count]['y_peak']
				x2 = peaks[count2]['x_peak']
				y2 = peaks[count2]['y_peak']
				distancex = (x1 - x2)**2
				distancey = (y1 - y2)**2
				distance = np.sqrt(distancex + distancey)

			#	if distance < 10:
			#		np.delete(peaks, count,0)
			#	else:
			#		peakssorted[count] = peak


			

		dlist = ["25"]
		dlistavg = 25
		loc = 0
		for count, peak in enumerate(peaks):
			#dlistavg = np.average(dlist)
			#print(peak)
			count2 = count + 1
			if count2 != len(peaks):
				x1 = peaks[count]['x_peak']
				y1 = peaks[count]['y_peak']
				x2 = peaks[count2]['x_peak']
				y2 = peaks[count2]['y_peak']
				#print(f"{x1},{y1} - {x2},{y2}")


				distancex = (x1 - x2)**2
				distancey = (y1 - y2)**2
				distance = np.sqrt(distancex + distancey)

				if distance < dlistavg + 10 and distance > dlistavg - 10:
					#dlist[loc] = distance
					#loc += 1
					print(f"Distance between {x1},{y1} and {x2},{y2} is: {distance}")


			#cv2.circle(np_pixel_map,tuple(peak),1,(0,0,255))

		#GD1 = interpolate.griddata((x1, y1), newarr.ravel(),(xx, yy),method='cubic')
		#psf = cv2.imread("psf.png", cv2.IMREAD_GRAYSCALE)

		cv2.imwrite(f"{filepathsave}/peaks.png", pixel_map)

			 
		print(f"finding peaks Finished!\n")
		return 
		
def find2(np_pixel_map, listofcoordstoinfill, width, height ,filepathsave):
		pixel_map = np_pixel_map.copy()

		from photutils.detection import find_peaks
		print(f"finding peaks Starting now...")

		mean = np_pixel_map.mean(axis=0).mean(axis=0)
		#print(f"Mean colour: {mean}")
		peaks = find_peaks(np_pixel_map, threshold=(mean+10))

		print(peaks)
		print(f"size before: {len(peaks)}")
		deletelist = []
		for count, peak in enumerate(peaks):
			count2 = count + 1
			if count2 != len(peaks):
				x1 = peaks[count]['x_peak']
				y1 = peaks[count]['y_peak']
				x2 = peaks[count2]['x_peak']
				y2 = peaks[count2]['y_peak']
				distancex = (x1 - x2)**2
				distancey = (y1 - y2)**2
				distance = np.sqrt(distancex + distancey)
				

				a = (x1 + 6, y1 + 6)

				b = (x2 - 6, y2 - 6)
				if distance < 5:
					if peaks[count]["peak_value"] > peaks[count2]["peak_value"]:
						print("Peaks count greater than count2")
						
					elif peaks[count]["peak_value"] == peaks[count2]["peak_value"]:
						print("Same value")
						cv2.rectangle(pixel_map, a, b, (0,0,255), 1)
					#cv2.line(pixel_map, (x1,y1), (x2,y2), (0,0,255), 1)

				#	deletelist[count-1] = True
			#	else:
					#deletelist[count-1] = False
					#print("less than 10.. Deleting!")
		#print(deletelist)
		#peakscleaned = np.delete(peaks, deletelist,0)



		#print(f"size after: {len(peakscleaned)}")
		cv2.imwrite(f"{filepathsave}/peaks.png", pixel_map)

			 
		print(f"finding peaks Finished!\n")
		return 

def find3(np_pixel_map, listofcoordstoinfill, width, height ,filepathsave):
		#pixel_map = np_pixel_map.copy()
		width, hieght = np_pixel_map.shape
		searcharea = 0
		a = (1,1)
		pixels = 1
		if width == 512:
				searcharea = 6
				a = (3,3)
				pixels = 1
				seperation = 12.5
		if width == 1024:
				searcharea = 8
				a = (6,6)
				pixels = 1
				seperation = 25
		if width == 2048:
				searcharea = 20
				a = (12,12)
				pixels = 2
				seperation = 50
		if width == 4096:
				searcharea = 50
				a = (24,24)
				pixels = 2
				seperation = 100
		if width == 8192:
				searcharea = 100
				a = (48,48)
				pixels = 4
				seperation = 200

		print(f"Search area set to: {searcharea}\nShape: {np_pixel_map.shape}")
		#pixel_map = cv2.cvtColor(np_pixel_map, cv2.COLOR_RGBA2GRAY)
		pixel_map = cv2.cvtColor(np_pixel_map,cv2.COLOR_GRAY2RGB)

		from photutils.detection import DAOStarFinder

		print(f"finding peaks Starting now...")

		mean = np_pixel_map.mean(axis=0).mean(axis=0)
		#print(f"Shape: {np_pixel_map.shape}")
		print(f"Mean colour as int: {int(mean)}")
		columns = DAOStarFinder(threshold=(int(mean)+10),fwhm=(searcharea),exclude_border=False)
		peaks = columns(np_pixel_map)
		fluxmean = np.mean(peaks['flux'])


		peaks_x_first = peaks["id","xcentroid","ycentroid","sharpness","roundness1","roundness2","flux"]
		peaks_x_first.sort("xcentroid")
		peaks_x_first["id"] = peaks["id"]

		peaks_y_first = peaks["id","xcentroid","ycentroid","sharpness","roundness1","roundness2","flux"]
		peaks_y_first.sort("ycentroid")
		peaks_y_first["id"] = peaks["id"]

		peaks = peaks_y_first

		shiftpercentage = 100.5

		for count, peak in enumerate(peaks):
			count2 = count + 1
			if count2 != len(peaks):
				roundvalue = peaks[count]['roundness1']
				roundvalue2 = peaks[count]['roundness2']
				fluxvalue = peaks[count]['flux']

				x1 = peaks[count]['xcentroid']
				y1 = peaks[count]['ycentroid']
				x2 = peaks[count2]['xcentroid']
				y2 = peaks[count2]['ycentroid']
				distancex = (x1 - x2)**2
				distancey = (y1 - y2)**2
				distance = np.sqrt(distancex + distancey)

				location = (peak['xcentroid'],peak['ycentroid'])
				location2 = (peaks[count2]['xcentroid'],peaks[count2]['ycentroid'])
				
				start = np.subtract(location,a)
				#print(f"Start: {tuple(np.int_(start))}")
				end = np.add(location,a)
				#print(f"End: {tuple(np.int_(end))}")
				cv2.rectangle(pixel_map, tuple(np.int_(start)), tuple(np.int_(end)), (0,0,100), pixels)

				if distance < seperation * 140/100:
					if distance > seperation * shiftpercentage/100:
						#print(f"UNUSUAL! Distance between {count} Peak and the next is: {distance}")
						cv2.line(pixel_map, tuple(np.int_(location)), tuple(np.int_(location2)), (0,255,0), pixels)
					elif distance < seperation * shiftpercentage/100:
						#print(f"UNUSUAL! Distance between {count} Peak and the next is: {distance}")
						cv2.line(pixel_map, tuple(np.int_(location)), tuple(np.int_(location2)), (0,0,255), pixels)

				
				if roundvalue2 > 0.5 or roundvalue2 < -0.5:
					print(colored(f"Roundness value: {roundvalue} || Roundness2 value: {roundvalue2}", "red"),colored(f"|| Flux Value: {fluxvalue}", "white"))
					#print(f"Roundness value: {roundvalue}")
					cv2.rectangle(pixel_map, tuple(np.int_(start)), tuple(np.int_(end)), (212,0,255), pixels)
				elif fluxvalue < (fluxmean * (85/100)):
					print(colored(f"Roundness value: {roundvalue} || Roundness2 value: {roundvalue2}", "white"),colored(f"|| Flux Value: {fluxvalue}", "red"))
					cv2.rectangle(pixel_map, tuple(np.int_(start)), tuple(np.int_(end)), (0,255,221), pixels)
				elif fluxvalue > (fluxmean * (115/100)):
					print(colored(f"Roundness value: {roundvalue} || Roundness2 value: {roundvalue2}", "white"),colored(f"|| Flux Value: {fluxvalue}", "blue"))
					cv2.rectangle(pixel_map, tuple(np.int_(start)), tuple(np.int_(end)), (221,255,0), pixels)

				#elif fluxvalue < (fluxmean - (fluxmean/3)):
    				#	print(colored(f"Roundness value: {roundvalue} || Roundness2 value: {roundvalue2}", "white"),colored(f"|| Flux Value: {fluxvalue}", "red"))
					#cv2.rectangle(pixel_map, tuple(np.int_(start)), tuple(np.int_(end)), (0,255,221), pixels)
				#elif fluxvalue > (fluxmean + (fluxmean/6)):
					#print(colored(f"Roundness value: {roundvalue} || Roundness2 value: {roundvalue2}", "white"),colored(f"|| Flux Value: {fluxvalue}", "blue"))
					#cv2.rectangle(pixel_map, tuple(np.int_(start)), tuple(np.int_(end)), (221,255,0), pixels)

					
		
		peaks = peaks_x_first

		for count, peak in enumerate(peaks):
			count2 = count + 1
			if count2 != len(peaks):
				roundvalue = peaks[count]['roundness1']
				roundvalue2 = peaks[count]['roundness2']
				fluxvalue = peaks[count]['flux']

				x1 = peaks[count]['xcentroid']
				y1 = peaks[count]['ycentroid']
				x2 = peaks[count2]['xcentroid']
				y2 = peaks[count2]['ycentroid']
				distancex = (x1 - x2)**2
				distancey = (y1 - y2)**2
				distance = np.sqrt(distancex + distancey)

				location = (peak['xcentroid'],peak['ycentroid'])
				location2 = (peaks[count2]['xcentroid'],peaks[count2]['ycentroid'])
				
				start = np.subtract(location,a)
				#print(f"Start: {tuple(np.int_(start))}")
				end = np.add(location,a)
				#print(f"End: {tuple(np.int_(end))}")
				
				if distance < seperation * 140/100:
					if distance > seperation * shiftpercentage/100:
						#print(f"UNUSUAL! Distance between {count} Peak and the next is: {distance}")
						cv2.line(pixel_map, tuple(np.int_(location)), tuple(np.int_(location2)), (0,255,0), pixels) # green = stretch
					elif distance < seperation * shiftpercentage/100:
						#print(f"UNUSUAL! Distance between {count} Peak and the next is: {distance}")
						cv2.line(pixel_map, tuple(np.int_(location)), tuple(np.int_(location2)), (0,0,255), pixels) # red = compress

		#print(peaks)
		num_columns_found = (np.array(peaks)).shape
		print(f"Number of Atomic columns detected: {num_columns_found}")
		print(f"Flux mean: {fluxvalue}")
		cv2.imwrite(f"{filepathsave}/peaks.png", pixel_map)

			 
		print(f"finding peaks Finished!\n Images saved to {filepathsave}")
		return 


def find4(np_pixel_map, listofcoordstoinfill, width, height ,filepathsave):
    		#pixel_map = np_pixel_map.copy()
		width, hieght = np_pixel_map.shape
		searcharea = 0
		a = (1,1)
		if width == 512:
				searcharea = 4
				a = (3,3)
				seperation = 12.5
		if width == 1024:
				searcharea = 8
				a = (6,6)
				seperation = 25
		if width == 2048:
				searcharea = 20
				a = (12,12)
				seperation = 50
		if width == 4096:
				searcharea = 40
				a = (24,24)
				seperation = 100
		if width == 8129:
				searcharea = 90
				a = (48,48)
				seperation = 200

		print(f"Search area set to: {searcharea}\nShape: {np_pixel_map.shape}")
		#pixel_map = cv2.cvtColor(np_pixel_map, cv2.COLOR_RGBA2GRAY)
		pixel_map = cv2.cvtColor(np_pixel_map,cv2.COLOR_GRAY2RGB)

		from photutils.detection import DAOStarFinder
		import operator

		
		print(f"finding peaks Starting now...")

		mean = np_pixel_map.mean(axis=0).mean(axis=0)
		print(f"Shape: {np_pixel_map.shape}")
		print(f"Mean colour as int: {int(mean)}")
		columns = DAOStarFinder(threshold=(int(mean)+10),fwhm=(searcharea),exclude_border=True)

		peaks = columns(np_pixel_map)
		fluxmean = np.mean(peaks['flux'])

		#np.array()
		print(np.array(peaks.as_array()))
		
		print("================================")
		#peaks_x_sort = (np.array(peaks.as_array()))
		

		#peaks_x_sort = (peaks.as_array()).sorted(key = operator.itemgetter(1))

		peaks_x_sort = np.array([(sub[1], int(sub[0]), sub[2],sub[3],sub[4],sub[5],sub[6],sub[7],sub[8],sub[9],sub[10]) for sub in peaks])
		#peaks_y_sort = np.array([(sub[2], int(sub[0]), sub[1],sub[3],sub[4],sub[5],sub[6],sub[7],sub[8],sub[9],sub[10]) for sub in peaks])
		print(peaks_x_sort.shape)
		peaks_x_sort = np.sort(peaks_x_sort, axis=0)
		print(peaks_x_sort)
		for count, peak in enumerate(peaks_x_sort):
			count2 = count + 1
			if count2 != len(peaks_x_sort):
				roundvalue = peaks_x_sort[count][4]
				roundvalue2 = peaks_x_sort[count][5]
				fluxvalue = peaks_x_sort[count][9]

				x1 = peaks_x_sort[count][1]
				y1 = peaks_x_sort[count][2]
				x2 = peaks_x_sort[count2][1]
				y2 = peaks_x_sort[count2][2]
				distancex = (x1 - x2)**2
				distancey = (y1 - y2)**2
				distance = np.sqrt(distancex + distancey)

				location = (x1,y1)
				location2 = (x2,y2)
				
				start = np.subtract(location,a)
				#print(f"Start: {tuple(np.int_(start))}")
				end = np.add(location,a)
				#print(f"End: {tuple(np.int_(end))}")
				
				if distance < seperation + 1:
					if distance > seperation:
						#print(f"UNUSUAL! Distance between {count} Peak and the next is: {distance}")
						cv2.line(pixel_map, tuple(np.int_(location)), tuple(np.int_(location2)), (0,255,0), 1)
					elif distance < seperation:
						#print(f"UNUSUAL! Distance between {count} Peak and the next is: {distance}")
						cv2.line(pixel_map, tuple(np.int_(location)), tuple(np.int_(location2)), (255,0,0), 1)
					else: 
						asdf = 1
						#print(f"NORMAL? Distance between {count} Peak and the next is: {distance}")
				cv2.rectangle(pixel_map, tuple(np.int_(start)), tuple(np.int_(end)), (0,0,100), 1)
				if roundvalue2 > 0.5 or roundvalue2 < -0.5:
					print(colored(f"Roundness value: {roundvalue} || Roundness2 value: {roundvalue2}", "red"),colored(f"|| Flux Value: {fluxvalue}", "white"))
					#print(f"Roundness value: {roundvalue}")
					cv2.rectangle(pixel_map, tuple(np.int_(start)), tuple(np.int_(end)), (212,0,255), 1)
				elif fluxvalue < (fluxmean - (fluxmean/4)):
					print(colored(f"Roundness value: {roundvalue} || Roundness2 value: {roundvalue2}", "white"),colored(f"|| Flux Value: {fluxvalue}", "red"))
					cv2.rectangle(pixel_map, tuple(np.int_(start)), tuple(np.int_(end)), (0,255,221), 1)
			
					
		

		#print(peaks_x_sort)
		print(f"shape of new list: {peaks_x_sort.shape}")
		print(f"Flux mean: {fluxvalue}")
		cv2.imwrite(f"{filepathsave}/peaks.png", pixel_map)

			 
		print(f"finding peaks Finished!\n Images saved to {filepathsave}")
		return



def v5(np_pixel_map, listofcoordstoinfill, width, height ,filepathsave):
		print(f"V5 Starting now...")
		#print(f"List of coords:\n{listofcoordstoinfill}")
		loc = listofcoordstoinfill

		search_loc = 1
		loca, locb = loc[search_loc,0], loc[search_loc,1]
		search_area = 3
		a = loca - search_area
		b = loca + search_area + 1
		c = locb - search_area
		d = locb + search_area + 1
		if a < 0:
			a = 0
		if b < 0:
			b = 0
		if c < 0:
			c = 0
		if d < 0:
			d = 0
		#print(f"test: {loca+1},{locb+1}\n Loca/b = {a}:{b}, {c}:{d})")
		test = np_pixel_map[(a):(b),(c):(d)]
		#print(f"Test found:\n{test}")		
		tarray = np_pixel_map
		tarray = tarray.astype('float')
		tarray[tarray == 0] = 'nan' # or use np.nan
		zerotofifty = np.arange(0, width, 1).tolist()
		fig = plt.figure()
		x = np.arange(0, tarray.shape[1])
		y = np.arange(0, tarray.shape[0])
		tarray[tarray==0] = np.nan
		tarray = np.ma.masked_invalid(tarray)
		xx, yy = np.meshgrid(x, y)
		x1 = xx[~tarray.mask]
		y1 = yy[~tarray.mask]
		newarr = tarray[~tarray.mask]

		GD1 = interpolate.griddata((x1, y1), newarr.ravel(),(xx, yy),method='cubic')
		plt.imshow(GD1, interpolation='bicubic')
		fig.savefig(f"{filepathsave}/graph.png")

		cv2.imwrite(f"{filepathsave}/Algorithm_processed.png", GD1)
		print(f"V5 Finished!\n")
		return GD1


def v6(np_pixel_map, listofcoordstoinfill, width, height ,filepathsave):
		print(f"V5 Starting now...")
		#print(f"List of coords:\n{listofcoordstoinfill}")
		loc = listofcoordstoinfill

		search_loc = 1
		loca, locb = loc[search_loc,0], loc[search_loc,1]
		search_area = 3
		a = loca - search_area
		b = loca + search_area + 1
		c = locb - search_area
		d = locb + search_area + 1
		if a < 0:
			a = 0
		if b < 0:
			b = 0
		if c < 0:
			c = 0
		if d < 0:
			d = 0
		#print(f"test: {loca+1},{locb+1}\n Loca/b = {a}:{b}, {c}:{d})")
		test = np_pixel_map[(a):(b),(c):(d)]
		#print(f"Test found:\n{test}")		
		tarray = np_pixel_map
		tarray = tarray.astype('float')
		tarray[tarray == 0] = 'nan' # or use np.nan
		zerotofifty = np.arange(0, width, 1).tolist()
		fig = plt.figure()
		x = np.arange(0, tarray.shape[1])
		y = np.arange(0, tarray.shape[0])
		tarray[tarray==0] = np.nan
		tarray = np.ma.masked_invalid(tarray)
		xx, yy = np.meshgrid(x, y)
		x1 = xx[~tarray.mask]
		y1 = yy[~tarray.mask]
		newarr = tarray[~tarray.mask]

		GD1 = interpolate.griddata((x1, y1), newarr.ravel(),(xx, yy),method='cubic')
		plt.imshow(GD1, interpolation='bicubic')
		fig.savefig(f"{filepathsave}/graph.png")

		cv2.imwrite(f"{filepathsave}/Algorithm_processed.png", GD1)
		print(f"V5 Finished!\n")
		return GD1