"""
Contains Action Detector class which loads an RNN model and,
passes the aggregated CNN features through it to predict an action
"""

import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

class PigActionDetector():

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
		action_names = ['explore', 'investigate']

		self.le = LabelEncoder()
		self.le.fit(action_names)

	def detect(self, cnn_embed_seq):
		"""
		Pass the CNN embedded sequence through the RNN
		"""
		with torch.no_grad():
			output = self.model(cnn_embed_seq)
			output = F.softmax(output)
			confidence, y_pred = output.topk(1, largest=True, sorted=True)

			return self.cat2labels(y_pred[0].tolist())[0], confidence[0].tolist()[0]

	def cat2labels(self, y_cat):
		"""
		Static method to convert categories to labels
		"""
		return self.le.inverse_transform(y_cat).tolist()
