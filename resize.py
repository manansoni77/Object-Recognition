import cv2
import os
import argparse
from imutils.paths import list_images

def load_images_from_folder(folder):
    images = []
    names = []
    for filename in os.listdir(folder):
        name = os.path.join(folder,filename)
        img = cv2.imread(name)
        if img is not None:
            images.append(img)
            names.append(name)
    return images,names

ap = argparse.ArgumentParser()
ap.add_argument("-p","--path",required=True,help="path to images")
args = vars(ap.parse_args())

images,names = load_images_from_folder("./"+args["path"])

for image,name in zip(images,names):
    shape = image.shape
    width = shape[1]
    height = shape[0]
    new_width = 500
    new_height = int((height*new_width)/width)
    new_dim = (new_width, new_height)
    resized = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(name, resized)
