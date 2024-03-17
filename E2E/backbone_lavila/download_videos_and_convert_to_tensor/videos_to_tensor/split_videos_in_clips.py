import subprocess

def split_video(input_file, output_prefix, duration=2):
    start_time = 0
    clip_number = 1

    while True:
        output_file = f"{output_prefix}_{clip_number}.mp4"

        # Run ffmpeg command to create a clip
        command = [
            'ffmpeg',
            '-ss', str(start_time),
            '-i', input_file,
            '-t', str(duration),
            '-c:v', 'copy',
            '-c:a', 'copy',
            output_file
        ]

        subprocess.run(command, capture_output=True)

        # Update start time for the next clip
        start_time += duration

        # Get video duration using ffprobe to check if the end of the video is reached
        ffprobe_command = [
            'ffprobe',
            '-i', input_file,
            '-show_entries', 'format=duration',
            '-v', 'quiet',
            '-of', 'csv=p=0'
        ]

        result = subprocess.run(ffprobe_command, capture_output=True, text=True)
        total_duration = float(result.stdout.strip())

        if start_time >= total_duration:
            break

        clip_number += 1

# Example usage:
input_file = 'P01_107.mp4'  # Replace with your input video file path
output_prefix = 'output_clip'   # Prefix for output clip filenames

split_video(input_file, output_prefix)