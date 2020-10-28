
print("Working...")
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

import cv2
import numpy as np
import argparse
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to test image")
args = vars(ap.parse_args())

model_architecture = 'architecture.json'
model_weights = 'weights.h5'

json_file = open('architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("weights.h5")

print("[INFO] Evaluating network...")
img = cv2.imread(args["image"])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))

predIdxs = model.predict(np.array([img,])/255.0, batch_size=1)[0]

if predIdxs[0] > predIdxs[1]:
	print("COVID-19 POSITIVE IMAGE")
else:
	print("NORMAL IMAGE")
	
