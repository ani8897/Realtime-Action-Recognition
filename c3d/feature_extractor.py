import os
import PIL
import numpy as np

from PIL import Image
from keras.models import Model
from keras.utils.data_utils import get_file

from c3d import C3D

C3D_MEAN_PATH = 'https://github.com/adamcasson/c3d/releases/download/v0.1/c3d_mean.npy'

class FeatureExtractor():
	"""docstring for FeatureExtractor"""
	
	def __init__(self, weights_path):

		c3d_model = C3D(weights_path)
		layer_name = 'fc6'
		self.model = Model(inputs=c3d_model.input, outputs=c3d_model.get_layer(layer_name).output)

	def extract(self, clip):
		"""
		Return extracted features
		"""
		return self.model.predict(clip)[0]

	@staticmethod
	def preprocess_clip(frames):
		"""Resize and subtract mean from clip input

		Keyword arguments:
		clip -- clip frames to preprocess. Expected shape
			(frames, rows, columns, channels). If the input has more than 16 frames
			then only 16 evenly samples frames will be selected to process.

		Returns:
		A numpy array.

		"""
		# clip = np.array(clip)
		# ## Redundant, remove later
		# intervals = np.ceil(np.linspace(0, clip.shape[0] - 1, 16)).astype(int)
		# frames = clip[intervals]

		# Reshape to 128x171
		reshape_frames = np.zeros((16, 128, 171, 3))
		frame_ids = [1 + i*3 for i in range(16)]
		for i, fid in enumerate(frame_ids):
			img = np.array(frames[fid].resize([171, 128], resample=PIL.Image.BICUBIC))
			reshape_frames[i, :, :, :] = img

		mean_path = get_file('c3d_mean.npy',
							 C3D_MEAN_PATH,
							 cache_subdir='models',
							 md5_hash='08a07d9761e76097985124d9e8b2fe34')

		# Subtract mean
		mean = np.load(mean_path)
		reshape_frames -= mean
		# Crop to 112x112
		reshape_frames = reshape_frames[:, 8:120, 30:142, :]
		# Add extra dimension for samples
		reshape_frames = np.expand_dims(reshape_frames, axis=0)

		return reshape_frames

	@staticmethod
	def load_clip(clip_path):
		# ## Redundant, remove later
		# intervals = np.ceil(np.linspace(0, clip.shape[0] - 1, 16)).astype(int)
		# frames = clip[intervals]
		frames = []
		for i in range(1,61):
			image = Image.open(os.path.join(clip_path, '%d.jpg'%i))
			frames.append(image)
		return frames

	def load_and_extract(self, clip_path):
		clip = self.load_clip(clip_path)
		processed_clip = self.preprocess_clip(clip)
		return self.extract(processed_clip)
