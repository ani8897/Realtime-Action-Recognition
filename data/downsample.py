import os
import cv2

import matplotlib.pyplot as plt 

def create_dir(dir_name):
	try: 
		os.stat(dir_name)
	except:
		os.mkdir(dir_name)

## Create directory to store all the compressed frames
FRAMES_DIR = 'action_frames'
COM_FRAMES_DIR = 'compressed_action_frames'
X_CROP, Y_CROP = 350,30

VISUALIZE = False

create_dir(COM_FRAMES_DIR)

## Go over all the videos in the folder
for video_name in os.listdir(FRAMES_DIR):
	print(video_name)
	VIDEO_DIR = os.path.join(FRAMES_DIR, video_name)

	## Go over all the action folders
	for action_path in os.listdir(VIDEO_DIR):
		ACTION_DIR = os.path.join(VIDEO_DIR, action_path)
		NEW_ACTION_DIR = os.path.join(COM_FRAMES_DIR, "%s>%s"%(video_name, action_path))
		create_dir(NEW_ACTION_DIR) ## Unique name to simplify training code

		## Go over all the frames for that action
		for frame_path in os.listdir(ACTION_DIR):
			frame = cv2.imread(os.path.join(ACTION_DIR, frame_path))

			h, w, _ = frame.shape

			## Crop the frame
			cropped_frame = frame[Y_CROP:h-Y_CROP, X_CROP:w-X_CROP]
			## Downsample by a factor of 4
			downsampled_frame = cv2.resize(cropped_frame, (0,0), fx=0.25, fy=0.25) 

			## Visualize for debugging
			if VISUALIZE:
				plt.figure()
				plt.imshow(cv2.cvtColor(downsampled_frame, cv2.COLOR_BGR2RGB))
				plt.show()
				input("")
				plt.close()

			cv2.imwrite('%s/%s'%(NEW_ACTION_DIR,frame_path), downsampled_frame)
			
