import os
import cv2
import pandas as pd

annotations_df = pd.read_csv('scoringproject_9_NOR.csv')
VIDEO_DIR = '../thesis/Edge-CV/dilger/keypoints/data/videos'
video_files = os.listdir(VIDEO_DIR)

def create_dir(dir_name):
	try: 
		os.stat(dir_name)
	except:
		os.mkdir(dir_name)

## Create directory to store all the frames
FRAMES_DIR = 'action_frames'
create_dir(FRAMES_DIR)

## Go over all the videos in the folder
for vf in video_files:
	video_name = vf.split('/')[-1].split('.')[0]
	video_stream = cv2.VideoCapture('%s/%s.mp4'%(VIDEO_DIR, video_name))

	## Create directory to store all frames per video
	print(video_name)
	create_dir(os.path.join(FRAMES_DIR, video_name))

	actions_df = annotations_df[annotations_df['Scoring'] == video_name]

	## Skip initial frames	
	assert actions_df.iloc[0]['Value'] == 'Start'

	start_frame, end_frame = int(actions_df.iloc[0]['Start_Frame']), int(actions_df.iloc[0]['Start_Frame'])
	[video_stream.read() for _ in range(start_frame)]
	actions_df = actions_df.iloc[1:]

	## Start annotating actions
	explore_count, investigate_count = 0, 0
	for index, row in actions_df.iterrows():

		action = row['Value']
		astart_frame = int(row['Start_Frame']) 
		## Annotate exploration
		if end_frame < astart_frame:
			dir_path = os.path.join(FRAMES_DIR, video_name, 'explore-%d[%d-%d]'%(explore_count, end_frame, astart_frame))
			explore_count += 1
			create_dir(dir_path)

			frame_id = 1
			for i in range(astart_frame - end_frame):
				_, frame = video_stream.read()
				cv2.imwrite('%s/%d.jpg'%(dir_path,frame_id), frame)
				frame_id += 1

		if action == 'End':
			break
		aend_frame = int(row['Stop_Frame'])

		action = action.split(' ')[0].lower()
		## Annotate investigation
		dir_path = os.path.join(FRAMES_DIR, video_name, 'investigate-%s-%d[%d-%d]'%(action, investigate_count,astart_frame,aend_frame))
		investigate_count += 1
		create_dir(dir_path)

		frame_id = 1
		for i in range(aend_frame - astart_frame):
			_, frame = video_stream.read()
			cv2.imwrite('%s/%d.jpg'%(dir_path,frame_id), frame)
			frame_id += 1

		start_frame, end_frame = astart_frame, aend_frame

