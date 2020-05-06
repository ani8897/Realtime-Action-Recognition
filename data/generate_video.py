import os
import cv2
from tqdm import tqdm

ROOT_DIR = 'compressed_action_frames-60-all'
CLIP_DIR = 'clip-compressed_action_frames-60-all'

if not os.path.exists(CLIP_DIR):
	os.mkdir(CLIP_DIR)

for video_name in tqdm(os.listdir(ROOT_DIR)):

	## Setup output video
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	clip_path = os.path.join(CLIP_DIR, '%s.mp4'%video_name) 
	out = cv2.VideoWriter(clip_path, fourcc, 30, (305, 255))

	for i in range(1,61):
		out.write(cv2.imread(os.path.join(ROOT_DIR, video_name, "%d.jpg"%i)))

	out.release()
