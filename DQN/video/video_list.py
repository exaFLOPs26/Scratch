import os
import subprocess

# Define paths
video_dir = os.path.join(os.path.dirname(__file__))
video_list_file = os.path.join(video_dir, 'video_list.txt')
output_video = os.path.join(video_dir, 'video.mp4')

# Delete the video file
if os.path.exists("DQN/video.video.mp4"):
    os.remove("DQN/video.video.mp4")
    print(f"Deleted video file")
else:
    print(f"Video file not found")

# Create the video list file
with open(video_list_file, 'w') as f:
    for video_file in sorted(os.listdir(video_dir)):
        if video_file.endswith('.mp4'):
            f.write(f"file '{os.path.join(video_dir, video_file)}'\n")

# Run ffmpeg to concatenate the videos
subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', video_list_file, '-c', 'copy', output_video])

# Delete the video file
if os.path.exists("DQN/video.video_list.txt"):
    os.remove("DQN/video.video_list.txt")
    print(f"Deleted video file")
else:
    print(f"Video file not found")