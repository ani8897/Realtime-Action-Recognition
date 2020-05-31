"""
Contains Action Detector class which loads an RNN model and,
passes the aggregated CNN features through it to predict an action
"""

import torch
import pickle
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

class ActionDetector():

	def __init__(self, rnn_model, rnn_checkpoint):
		"""
		Initialize the detector which uses LRCN model.
		Since the CNN forward pass is done on the Raspberry Pi,
		we need to aggregate the features and pass them through an RNN
		"""
		rnn_model.load_state_dict(torch.load(rnn_checkpoint))
		self.model = rnn_model
		## Never forget to set the model in eval mode, else BatchNorm creates a problem
		self.model.eval()  

		## Convert categories to labels
		with open('resources/UCF101actions.pkl', 'rb') as f:
		    action_names = pickle.load(f)   # load UCF101 actions names

		self.le = LabelEncoder()
		self.le.fit(action_names)

	def detect(self, feature_bag):
		"""
		Stack the CNN features and pass through the RNN
		"""
		cnn_embed_seq = torch.stack(feature_bag, dim=0).unsqueeze(0) ## Append batch dimension
		
		with torch.no_grad():
			output = self.model(cnn_embed_seq)
			# y_pred = output.max(1, keepdim=True)[1] ## To output the max label
			confidence, y_pred = output.topk(5, largest=True, sorted=True)
			print(y_pred, confidence)
			return self.cat2labels(y_pred[0].tolist()), confidence[0].tolist()

	def cat2labels(self, y_cat):
		"""
		Static method to convert categories to labels
		"""
		return self.le.inverse_transform(y_cat).tolist()
	
