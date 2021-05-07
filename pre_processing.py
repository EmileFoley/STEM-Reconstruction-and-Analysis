import cv2
import numpy as np

print("pre_processing.py loaded")



def apply(np_pixel_map, width, height ,filepathsave):
		print(f"\npre_processing starting...")
		pre_processed_np_pixel_map = np_pixel_map.copy()
		cv2.fastNlMeansDenoising(np_pixel_map, pre_processed_np_pixel_map, 31, 7, 21)
		cv2.imwrite(f"{filepathsave}/Pre_Processed.png", pre_processed_np_pixel_map)
		print(f"pre_processing Finished!\n")
		return pre_processed_np_pixel_map
