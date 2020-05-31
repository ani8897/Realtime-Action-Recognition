"""
Dataset class for training LRCN
"""

import os
import torch
from torch.utils import data

from PIL import Image

class Dataset_CRNN(data.Dataset):
	
	def __init__(self, root_path, folders, labels, transform=None):
		self.root_path = root_path
		self.labels = labels
		self.folders = folders
		self.transform = transform

	def __len__(self):
		return len(self.folders)

	def read_images(self, selected_folder):
		X = []
		for i in range(1,61):
			image = Image.open(os.path.join(self.root_path, selected_folder, '%d.jpg'%i))
			if self.transform is not None:
				image = self.transform(image)

			X.append(image)

		return torch.stack(X, dim=0)

	def __getitem__(self, index):

		X = self.read_images(self.folders[index])     
		y = torch.LongTensor([self.labels[index]]) # (labels) LongTensor are for int64 instead of FloatTensor

		return X, y
