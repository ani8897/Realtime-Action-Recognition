"""
Main file containing primary logic to process frames and display as a web application
"""

## Install imagezmq from https://github.com/jeffbass/imagezmq
import os
import sys
import time
import socket
import signal
import threading
sys.path.append('/opt/imagezmq/imagezmq') 
sys.path.append('../LRCN')

import imagezmq
from manager import FrameManager
from detector import ActionDetector

from flask import Flask, render_template, Response

app = Flask("Action Recognition Dashboard")

## Frame Manager
MANAGER = None
DONE = False

def signal_handler(sig, frame):
	print('Exiting program')
	DONE = True
	sys.exit(0)

@app.route('/')
def index():
	return render_template('index.html')

def display_frame():
	"""
	Display frames processed by the Frame Manager
	"""
	global MANAGER

	while not DONE:
		byte_frame = MANAGER.emit()
		time.sleep(9./30)  
		yield (b'--frame\r\n'
			   b'Content-Type: image/jpeg\r\n\r\n' + byte_frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
	return Response(display_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

def consume_frames():
	"""
	Receive frames from Raspberry Pi
	"""
	global MANAGER

	## Setting socket to receive frames
	SOCK = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	SOCK.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	server_address = ('0.0.0.0', 8888)
	SOCK.bind(server_address)

	IMAGE_HUB = imagezmq.ImageHub()
	while not DONE:
		## Receive frame
		rpi_name, frame = IMAGE_HUB.recv_image()
		IMAGE_HUB.send_reply(b'OK')
		
		## Receive CNN features
		rpi_name, features = IMAGE_HUB.recv_image()
		IMAGE_HUB.send_reply(b'OK')

		## Frame manager consumes the frame for further processing
		MANAGER.consume(frame, features)

if __name__ == '__main__':

	## Initialize frame manager
	from model import DecoderRNN
	rnn_decoder = DecoderRNN(CNN_out=512, h_RNN_layers=3, h_RNN=512, h_FC_dim=256, dropout=0, num_classes=101)
	action_detector = ActionDetector(rnn_decoder,'../checkpoints/rnn-ucf101.pth')
	MANAGER = FrameManager(detector=action_detector)

	## Registering signal handler to exit the program gracefully
	signal.signal(signal.SIGINT, signal_handler)

	threading.Thread(target=app.run, args=('0.0.0.0',)).start()
	consume_frames()