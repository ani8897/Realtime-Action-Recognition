import sys
sys.path.append('../..')
from utils import cat2labels

import torch
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

class ActionDetector():

	def __init__(self, rnn_model, rnn_checkpoint):
		rnn_model.load_state_dict(torch.load(rnn_checkpoint))
		self.model = rnn_model

		with open('../base_code/UCF101actions.pkl', 'rb') as f:
		    action_names = pickle.load(f)   # load UCF101 actions names

		# convert labels -> category
		self.le = LabelEncoder()
		self.le.fit(action_names)

	def detect(self, feature_bag):
		cnn_embed_seq = torch.stack(feature_bag, dim=0).unsqueeze(0) ## Append batch dimension
		
		with torch.no_grad():
			output = self.model(cnn_embed_seq)
			# y_pred = output.max(1, keepdim=True)[1]
			confidence, y_pred = output.topk(5, largest=True, sorted=True)
			print(y_pred, confidence)
			return cat2labels(self.le, y_pred[0].tolist()), confidence[0].tolist()
	