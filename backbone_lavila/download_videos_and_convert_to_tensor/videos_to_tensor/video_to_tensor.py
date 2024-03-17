import sys
import os
import subprocess

import av
import numpy as np
import torch

from split_videos.split_videos_in_jpg import split_video_to_jpg
from image_to_tensor.image_to_tensor_h5 import images_to_hdf5


def convert_videos_to_jpg(EPIC_KITCHENS_VIDEO_DIR, EPIC_KITCHENS_IMAGE_DIR):
    '''
    creates folders for extracting jpg from each participant's video
    and then extract jpgs
    '''
    for idx, participant_dir in enumerate(os.listdir(EPIC_KITCHENS_VIDEO_DIR)):
        participant_image_dir = os.path.join(EPIC_KITCHENS_IMAGE_DIR, participant_dir)
        print(participant_image_dir)

        if not os.path.exists(participant_image_dir):
            os.makedirs(participant_image_dir)
        
        participant_video_dir = os.path.join(EPIC_KITCHENS_VIDEO_DIR, participant_dir)

        for _idx, video in enumerate(os.listdir(participant_video_dir)):

            video_image_dir = os.path.join(participant_image_dir, f'{participant_dir}_{_idx}')
            if not os.path.exists(video_image_dir):
                os.makedirs(video_image_dir)

            split_video_to_jpg(os.path.join(participant_video_dir, video), video_image_dir)
            print(f'finished converting in jpg video {idx} of participant {participant_dir}')


def convert_jpg_to_tensor(EPIC_KITCHENS_VIDEO_DIR, EPIC_KITCHENS_IMAGE_DIR, EPIC_KITCHENS_TENSOR_DIR):
    '''
    creates folder for extracting tensor for each participant
    and then convert each jpg of each participant's video in an tensor of shape (t,h,w,c)
    '''

    for idx, participant_dir in enumerate(os.listdir(EPIC_KITCHENS_IMAGE_DIR)):
        participant_tensor_dir = os.path.join(EPIC_KITCHENS_TENSOR_DIR, participant_dir)
        print(participant_tensor_dir)

        if not os.path.exists(participant_tensor_dir):
            os.makedirs(participant_tensor_dir)

        images_to_hdf5(os.path.join(EPIC_KITCHENS_IMAGE_DIR, participant_dir), f'{participant_tensor_dir}\{idx}.h5')
        print(f'finished converting in tensor video {idx} of participant {participant_dir}')


