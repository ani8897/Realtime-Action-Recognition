import os
import cv2
import torch

from app.detector import PigActionDetector
from model import ResnetEncoder, DecoderRNN
from pi.feature_extractor_cpu import FeatureExtractorCPU


## Load CNN encoder
cnn_encoder = ResnetEncoder(fc1_=1024, fc2_=1024, dropout=0.0, CNN_out=512)
extractor = FeatureExtractorCPU(cnn_encoder, 'checkpoints/cnn_encoder_epoch21.pth')

## Load RNN decoder
rnn_decoder = DecoderRNN(CNN_out=512, h_RNN_layers=3, h_RNN=64, h_FC_dim=16, dropout=0, num_classes=2)
action_detector = PigActionDetector(rnn_decoder, 'checkpoints/rnn_decoder_epoch21.pth')

base_dir = 'data/videos/'
video_list = os.listdir(base_dir)
print("Processing %d videos"%len(video_list))

for video_p in video_list:
	print(video_p)

	## Setup reading from video stream
	video_path = os.path.join(base_dir, video_p)
	video_stream = cv2.VideoCapture(video_path)

	## Setup output video
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	video_name = video_path.split('/')[-1].split('.')[0]
	out = cv2.VideoWriter(video_name+'-annotated.mp4', fourcc, 30, (1920, 1080))

	frame_id, frame_buffer = 0, []
	with torch.no_grad():
		while True:
			ret, frame = video_stream.read()
			if frame is None: break
			height, width, _ = frame.shape
			frame_id+=1

			frame_buffer.append(frame)

			if len(frame_buffer) == 30:				
				processed_frames = extractor.preprocess_frames(frame_buffer)
				prediction, score = action_detector.detect(extractor.extract(processed_frames))

				for f in frame_buffer:
					cv2.putText(f,"%s: %f"%(prediction, score),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
					out.write(f)

					cv2.imshow("Action Recognition model", frame)

				frame_buffer = []

			key = cv2.waitKey(1) & 0xFF
			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break

	out.release()
	cv2.destroyAllWindows()
