# import required packages
from imutils.paths import list_images
import numpy as np
import cv2
import argparse


# construct the cmd line argument parser
# 4 arguments -> image, prototxt file, DL model, confidence
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--images', required=True, help="Enter path to image")
ap.add_argument('-p', '--model-structure', required=True, help="Enter path to prototxt file")
ap.add_argument('-m', '--model-weights', required=True, help="Enter path to caffe model file")
ap.add_argument('-c', '--confidence', type=float, default=0.1, help="Enter minimum confidence")
args = vars(ap.parse_args())


# Classes of objects our pretrained model can recognise
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]


# nadomly generating colors for bounding boxes
np.random.seed(100)
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# load the pretrained model
print('[X] Loading Model')
net = cv2.dnn.readNetFromCaffe(args['model_structure'], args['model_weights'])


# list images
imagepaths = list_images(args['images'])
images = []
for imagepath in imagepaths:
	image = cv2.imread(imagepath)
	images.append(image)


# Detect objects in images
for image in images:
	# extract dimensions of image and convert to a blob of 300x300
	(h,w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 0.007843, (300,300), 127.5)
	# pass blob through the network to get detections
	print('[X] Detecting Objects ....')
	net.setInput(blob)
	detections = net.forward()
	# interpretting detections and displaying result
	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0,0,i,2]
		if confidence>args['confidence']:
			idx = int(detections[0,0,i,1])
			box = detections[0,0,i,3:7] * np.array([w,h,w,h])
			(startx, starty, endx, endy) = box.astype('int')
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			print("[INFO] {}".format(label))
			cv2.rectangle(image, (startx, starty), (endx, endy), COLORS[idx], 2)
			y = starty - 15 if starty - 15 > 15 else starty + 15
			cv2.putText(image, label, (startx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			cv2.imshow("output", image)
			cv2.waitKey(0)

