import os
from sklearn.model_selection import train_test_split

ROOT_DIR = '/home/aniket/Desktop/Livestock-Action-Recognition/data/flow-compressed_action_frames-60-all'

X, y = [], []
for video_name in os.listdir(ROOT_DIR):

	if 'explore' in video_name:
		X.append("%s %d %s"%(os.path.join(ROOT_DIR, video_name), 60, 'explore'))
		y.append('explore')

	if 'investigate' in video_name:
		X.append("%s %d %s"%(os.path.join(ROOT_DIR, video_name), 60, 'investigate'))
		y.append('investigate')

train_X, val_X, _, _ = train_test_split(X, y, test_size=0.25, random_state=42)

for t in train_X:
	print(t)

for _ in range(10):
	print()

for t in val_X:
	print(t)