# Realtime-Action-Recognition

The video demonstration for the project can be found [here](https://youtu.be/eeVDUKJaNkg)

<img src="results/result.gif" width="400"></img>

This repo is divided into nine directories:
1. ``app``: Contains Flask application used for real-time web application
2. ``pi``: Contains code for communicating frames and extracting CNN features from Intel's Neural Compute Stick
3. ``data``: Contains code for preprocessing dataset
4. ``references``: Contains the references used in our project
5. ``results``: Contains all the plots and code for generating the evaluation metrics
6. ``ncs``: Contains code to convert Pytorch model to Neural Compute Stick binary
7. ``LRCN``: Contains training and annotation code for LRCN
8. ``C3D``: Contains training and annotation code for C3D
9. ``TSM``: Contains training and annotation code for TSM

## Dataset

Download the dataset and the annotations from [this](https://uofi.box.com/s/1tihqo6sxwh1f0g6413rw12ij0o8tnk0) box link and place under the ``data`` folder. 
Use the scripts [extract_frames.py](data/extract_frames.py) and [downsample.py](data/downsample.py) to generate the annotation frames from the dataset.

## Model Weights

Download the ``checkpoints`` folder from [here](https://drive.google.com/drive/folders/1o30bqJo_OPxwVgMH_ChI2iN7sLvJo83l?usp=sharing) to reproduce our results.

To run our action recognition model on Intel's Movidius stick, download the pretrained binaries folder``ncs_models`` folder from [here](https://drive.google.com/drive/folders/1o30bqJo_OPxwVgMH_ChI2iN7sLvJo83l?usp=sharing) and place it inside the ``ncs`` folder.

## LRCN 

Run ``python3 train.py`` to train the model. Run ``python3 annotate.py`` to annotate the video dataset

## C3D

Precompute C3D features using the script [extract.py](C3D/extract.py). Run ``python3 train.py`` to train the model. Run ``python3 annotate-folder.py`` to annotate the video dataset.

## TSM

Follow the procedure specified [here](https://github.com/mit-han-lab/temporal-shift-module) to generate the dataset. Run the following command to train the model:
```
python3 main.py pig RGB \
      -p 2 --arch resnet18 --num_segments 8  --gd 20 --lr 0.02 \
      --wd 1e-4 --lr_steps 12 25 --epochs 35 --batch-size 64 -j 16 --dropout 0.5 \
      --consensus_type=avg --eval-freq=1 --shift --shift_div=8 --shift_place=blockres --npb
``` 
Run ``python3 annotate.py`` to annotate the video dataset.

## Neural Compute Stick

Run ``python3 pytorch_to_tf.py`` to convert the a Pytorch checkpoint into an onnx file. Install Intel's Model Optimizer to convert this onnx file into an NCS binary

## Pi

Run ``python3 send_frames.py -i <Host IP>`` to communicate frames from Raspberry Pi to Host laptop

## app

Run ``python3 main.py`` to launch the Flask application.
