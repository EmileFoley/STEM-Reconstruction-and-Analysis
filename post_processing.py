import cv2
import numpy as np

print("post_processing.py loaded")

def apply(np_pixel_map, width, height ,filepathsave):
	print(f"\npost_processing starting...")
	post_processed_np_pixel_map = np_pixel_map.copy()
	np_pixel_map = np.uint8(post_processed_np_pixel_map)
	#post_processed_np_pixel_map = cv2.fastNlMeansDenoisingMulti(np_pixel_map, 2, 5, None, 4, 7, 35)
	post_processed_np_pixel_map = cv2.fastNlMeansDenoising(np_pixel_map, post_processed_np_pixel_map, 31, 7, 21)
	cv2.imwrite(f"{filepathsave}/Post_Processed.png", post_processed_np_pixel_map)
	print(f"post_processing Finished!\n")
	return post_processed_np_pixel_map
