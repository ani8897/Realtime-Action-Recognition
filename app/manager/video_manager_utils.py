
from manager.parameters import *
import cv2
import numpy as np

def get_video_clips(video_path):
    frames = get_video_frames(video_path)
    clips = sliding_window(frames, CLIP_SIZE, BAG_SIZE)
    return clips, len(frames)

def sliding_window(arr, size, stride):
    num_chunks = int((len(arr) - size) / stride) + 2
    result = []
    for i in range(0,  num_chunks * stride, stride):
        if len(arr[i:i + size]) > 0:
            result.append(arr[i:i + size])
    return np.array(result)


def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            break
    return frames


    
