from video_to_tensor import convert_videos_to_jpg, convert_jpg_to_tensor
import time

EPIC_KITCHENS_VIDEO_DIR = './videos'
EPIC_KITCHENS_IMAGE_DIR = './images'
EPIC_KITCHENS_TENSOR_DIR = './tensor'

start_time = time.time()

print(f'start time is: {start_time}')

convert_videos_to_jpg(EPIC_KITCHENS_VIDEO_DIR, EPIC_KITCHENS_IMAGE_DIR)

convert_jpg_to_tensor(EPIC_KITCHENS_VIDEO_DIR, EPIC_KITCHENS_IMAGE_DIR, EPIC_KITCHENS_TENSOR_DIR)

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")