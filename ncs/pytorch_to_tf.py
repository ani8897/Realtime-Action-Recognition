## Import onnx before pytorch, else you will get a Segmentation Fault!
import onnx
from onnx_tf.backend import prepare

import os
import sys
import tensorflow as tf

import torch
from torch.autograd import Variable

sys.path.insert(0, '../../video-classification-master/ResNetCRNN/')
from functions import *

checkpoint = '../checkpoints/cnn_encoder_epoch28.pth'

# Load model checkpoint that is to be evaluated
state_dict = torch.load(checkpoint)
model = ResCNNEncoderPi(fc_hidden1=1024, fc_hidden2=768, drop_p=0, CNN_embed_dim=512)
model.load_state_dict(state_dict)

dummy_input = Variable(torch.randn(1, 3, 224, 224)) # nchw
onnx_filename = os.path.split(checkpoint)[-1].split('.')[0] + ".onnx"
torch.onnx.export(model, dummy_input, onnx_filename, verbose=True)