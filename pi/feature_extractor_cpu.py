import cv2
import torch
import torchvision.transforms as transforms

from PIL import Image

X_CROP, Y_CROP = 350,30

class FeatureExtractorCPU():
	
	def __init__(self, cnn_model, cnn_checkpoint):
		cnn_model.load_state_dict(torch.load(cnn_checkpoint))
		self.model = cnn_model
		self.model.eval()

		self.transform = transforms.Compose([transforms.Resize([224, 224]),
						transforms.ToTensor(),
						transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

	def extract(self, processed_frames):
		"""
		Return extracted features
		"""
		return self.model(processed_frames)

	def preprocess_frames(self, frame_buffer):

		frame_stack = []
		for f in frame_buffer:
			h, w, _ = f.shape
			## Crop the frame
			cropped_frame = f[Y_CROP:h-Y_CROP, X_CROP:w-X_CROP]
			## Downsample by a factor of 4
			downsampled_frame = cv2.resize(cropped_frame, (0,0), fx=0.25, fy=0.25) 
			frame_stack.append(self.transform(Image.fromarray(downsampled_frame)))

		return torch.stack(frame_stack, dim=0).unsqueeze(0)