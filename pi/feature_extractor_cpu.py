import cv2
import torch
import torchvision.transforms as transforms

import numpy as np
from PIL import Image

X_CROP, Y_CROP = 350,30

class FeatureExtractorCPU():
	
	def __init__(self, cnn_model, cnn_checkpoint):
		cnn_model.load_state_dict(torch.load(cnn_checkpoint))
		self.model = cnn_model
		self.model.eval()

		self.mean = [0.485, 0.456, 0.406]
		self.std = [0.229, 0.224, 0.225]

		self.transform = transforms.Compose([
						# transforms.Resize([224, 224]),
						transforms.ToTensor(),
						transforms.Normalize(mean=self.mean, std=self.std)])

	def extract(self, processed_frames):
		"""
		Return extracted features
		"""
		return self.model(processed_frames)

	def preprocess_frames(self, frame_buffer):

		frame_stack = []
		for f in frame_buffer:
			h, w, _ = f.shape
			f = f / 255.0
			## Crop the frame
			cropped_frame = f[Y_CROP:h-Y_CROP, X_CROP:w-X_CROP]
			## Downsample by a factor of 4
			downsampled_frame = cv2.resize(cropped_frame, (0,0), fx=0.25, fy=0.25) 

			frame = cv2.resize(downsampled_frame, (224, 224))
			cv2.imshow("Downsampled frame", frame)

			frame = frame.astype('float32')

			frame_stack.append(self.transform(frame))
		
			# frame_stack.append(self.transform(Image.fromarray(np.uint8(downsampled_frame))))

		return torch.stack(frame_stack, dim=0).unsqueeze(0)