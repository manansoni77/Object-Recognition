# TSF
TSF Internship Submissions

Description
This Project uses a pretrained model (MobileNetSSD) to recognise objects in images

Object_Detection.py - Script that loads images and models from paths specified and detects objects in the loaded images
Resize.py - A functional script you might wanna use to resize images in the Photos directory, not required but makes viewing image pop-ups easy
MobileNetSSD.prototxt - Stores the structure of the Neural Net used here
MobileNetSSD.caffemodel - Stores the weights of the Neural Net used here
Photos directory stores some sample pictures which I have used to demonstrate this project

The project is built with Python 3.8
If you want to run this project in your PC, you will need some specific libraries, namely cv2 (version 2), imutils, argparse and numpy

Commands to run this project on your machine
* Firstly open cmd and navigate to folder containing all files
* To resize images in Photos directory "python resize.py --path Photos"
* To detect objects in Photos directory "python Object_Detection.py --image Photos --model-weights MobileNetSSD.caffemodel --model-structure MobileNetSSD.prototxt.txt -c 0.5"

Every image will be displayed like a pop up with bounding boxes, class, and confidence (in percentage) for every detected object, a list of detected objects with confidence level will also be printed to console
