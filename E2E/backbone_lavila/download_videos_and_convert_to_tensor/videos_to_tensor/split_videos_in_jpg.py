import os 
import subprocess

def split_video_to_jpg(input_video, output_directory, fps= 30):
    '''
    Split a video into JPEG frames and write them to disk.
    Args:
        input_video (str): Path to the input video file.
        output_directory (str): Directory to save the extracted JPEG frames.
        fps (int): Frames per second for extracting frames (default is 24).
    '''
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_pattern = os.path.join(output_directory, 'frame_%04d.jpg')
    command = f"ffmpeg -i {input_video} -vf fps={fps} {output_pattern}"

    subprocess.run(command, shell=True)
