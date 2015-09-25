from detectors.face import FaceDetector
from picamera.array import PiArrayOutput
from picamera import PiCamera
from sklearn.svm import LinearSVC
from PIL import Image
import RPi.GPIO as gpio
import numpy as np
import argparse
import time
import cv2
import os
import io

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--face', required = True, help = 'path to face cascade')
args = vars(ap.parse_args())

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
raw_capture = PiArrayOutput(camera, size=(640, 480))

happy = 18
sad = 15

gpio.setmode(gpio.BCM)
gpio.setwarnings(False)
gpio.setup(happy, gpio.OUT)
gpio.setup(sad, gpio.OUT)

fd = FaceDetector(args['face'])
faceCascade = cv2.CascadeClassifier(args['face'])
time.sleep(0.1)

print('Analyzing faces...')

path = '../yalefaces'
img_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.happy') or f.endswith('.sad') or f.endswith('.normal')]
exp = dict(sad = 0, normal = 0, happy = 1)
imgs = []; labels = []

for img_path in img_paths:
    img_pil = Image.open(img_path).convert('L')
    image = np.array(img_pil)
    label = exp[os.path.split(img_path)[1].split('.')[1]]
    faces = faceCascade.detectMultiScale(image)
    for (x, y, w, h) in faces:
        imgs.append(image[y: y + 150, x: x + 150].flatten())
        labels.append(label)

clf = LinearSVC()
clf.fit(imgs, labels)

print('Finished learning')

stream = io.BytesIO()
for f in camera.capture_continuous(stream, format='jpeg'):
     stream.seek(0)
     image = Image.open(stream).convert('L')
     arr = np.array(image)
     faces = faceCascade.detectMultiScale(arr)
     print(faces)

     for (x, y, w, h) in faces:
         exp = clf.predict(arr[y: y + 150, x: x + 150].flatten())
         if exp:
             print('Happy')
             gpio.output(sad, 0)
             gpio.output(happy, 1)
         else:
             print('Sad/Neutral')
             gpio.output(happy, 0)
             gpio.output(sad, 1)

     stream.seek(0)
     stream.truncate()
