"""
Contains Detector class which applies a pretrained binary classifier from the extracted C3D features
"""
import torch
import torch.nn.functional as F

class C3D_detector():

	def __init__(self, model, checkpoint):
		"""
		Load binary classifier from checkpoint
		"""
		model.load_state_dict(torch.load(checkpoint))
		self.model = model
		self.model.eval()

		self.action_names = ['explore', 'investigate']

	def detect(self, c3d_feature):
		"""
		Given a C3D feature vector for a clip, detect the action
		"""
		out = self.model(torch.from_numpy(c3d_feature))

		score, y_pred = F.softmax(out).max(dim=0)
		
		return self.action_names[y_pred], float(score)