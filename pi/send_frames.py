"""
Script running on Raspberry Pi to send video frames to Cloud
"""

import sys
import signal
import argparse
import numpy as np

## Install imagezmq from https://github.com/jeffbass/imagezmq
sys.path.append('/opt/imagezmq/imagezmq/')

import cv2
import socket
import imagezmq

from picamera import PiCamera
from picamera.array import PiRGBArray

from feature_extractor import FeatureExtractor

def signal_handler(sig, frame):
	sys.exit(0)

def send_frames(size, host_ip):
	
	extractor = FeatureExtractor('../ncs/ncs_models/ucf101-resnet18.bin', '../ncs/ncs_models/ucf101-resnet18.xml')
	
	video_capture = cv2.VideoCapture(0)
	sender = imagezmq.ImageSender(connect_to='tcp://%s:5555'%host_ip)

	while True:
		ret, frame = video_capture.read()
		# frame = np.copy(f.array)
		sender.send_image(socket.gethostname(), frame)
		frame = extractor.preprocess_frame(frame)
		features = extractor.extract(frame)
		sender.send_image(socket.gethostname(), features['218'])

		if cv2.waitKey(1) == ord('q'):
			break
	
	video_capture.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--host_ip", help="IP address of the host to which the frames are sent over Ethernet", required=True)
	args = parser.parse_args()

	## Registering signal handler to exit the program gracefully
	signal.signal(signal.SIGINT, signal_handler)

	size = (640, 640)
	send_frames(size, args.host_ip)
