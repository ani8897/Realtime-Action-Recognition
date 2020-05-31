"""
Class to handle frames received from Raspberry Pi and apply action recognition model to it
"""

import cv2
import time
import torch
import numpy as np
from queue import Queue

CLIP_SIZE = 30

class FrameManager():
	def __init__(self, detector, debug=False):
		self.detector = detector
		self.debug = debug

		self.frame_counter = 0
		self.clip_counter = 0

		self.clip = []
		self.feature_bag = []

		self.emit_queue = Queue()

		## Debug
		self.t_ = None

	def populate_emit_queue(self, prediction, probability):
		"""
		Assign action prediction to each frame and insert cv2 frame into the emit queue
		"""

		for frame in self.clip:
			for i, p in enumerate(prediction):
				cv2.putText(frame,"%s: %f"%(p, probability[i]),(30,30+i*20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2,cv2.LINE_AA)
			self.emit_queue.put(frame)	

	def predict_and_reset_clip(self):
		"""
		Predict action for the clip and then reset it
		"""
		prediction, probability = self.detector.detect(self.feature_bag)
		self.populate_emit_queue(prediction, probability)

		self.clip = []
		self.feature_bag = []
		self.frame_counter = 0

	def consume(self, frame, features):
		"""
		Consumes and buffers Raspberry Pi frames
		"""
		if self.frame_counter == 0 and self.clip_counter == 0:
			if self.t_ is not None:
				print("Time to process the clip: %f"%(time.time() - self.t_))
			self.t_ = time.time()

		self.clip.append(frame)
		self.feature_bag.append(torch.from_numpy(np.array(features[0])))
		self.frame_counter += 1

		if self.frame_counter == CLIP_SIZE:
			self.predict_and_reset_clip()	

	def emit(self):
		"""
		Emits a single frame (in bytes) to Flask for visualization 
		"""
		frame = self.emit_queue.get()
		return cv2.imencode('.jpg', frame)[1].tobytes()
		