import os
import subprocess

# Define paths
video_dir = os.path.join(os.path.dirname(__file__))
video_list_file = os.path.join(video_dir, 'video_list.txt')
output_video = os.path.join(video_dir, 'video.mp4')

# Delete the existing output video file if it exists
if os.path.exists(output_video):
    os.remove(output_video)
    print(f"Deleted existing output video file: {output_video}")
else:
    print(f"Output video file not found: {output_video}")

# Create the video list file
with open(video_list_file, 'w') as f:
    video_files = []
    for video_file in sorted(os.listdir(video_dir)):
        if video_file.endswith('.mp4') and video_file != 'video.mp4':
            video_path = os.path.join(video_dir, video_file)
            f.write(f"file '{video_path}'\n")
            video_files.append(video_path)

# Run ffmpeg to concatenate the videos
subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', video_list_file, '-c', 'copy', output_video])

# Delete the video list file
if os.path.exists(video_list_file):
    os.remove(video_list_file)
    print(f"Deleted video list file: {video_list_file}")
else:
    print(f"Video list file not found: {video_list_file}")

# Delete all the video files that were in the list file
for video_file in video_files:
    if os.path.exists(video_file):
        os.remove(video_file)
        print(f"Deleted video file: {video_file}")
    else:
        print(f"Video file not found: {video_file}")