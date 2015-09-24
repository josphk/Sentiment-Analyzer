from detectors.face import FaceDetector
from imgtools import imgtools
from picamera.array import PiArrayOutput
from picamera import PiCamera
from PIL import Image
from sklearn.svm import LinearSVC
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

scale_factor = input('Enter scale factor: ')
min_neighbours = input('Enter minimum neighbours: ')

stream = io.BytesIO()
for f in camera.capture_continuous(stream, format='jpeg'):
     stream.seek(0)
     image = Image.open(stream).convert('L')
     arr = np.array(image)
     faces = faceCascade.detectMultiScale(arr)
     print(faces)

     for (x, y, w, h) in faces:
         exp = clf.predict(arr[y: y + 150, x: x + 150].flatten())
         if exp == 1:
             print('Happy')
         else:
             print('Sad/Neutral')

     stream.seek(0)
     stream.truncate()

#for f in camera.capture_continuous(raw_capture, format='bgr', use_video_port = True):
#    frame = Image.open(f).convert('L')
#    fra = np.array(frame)
#    print(fra)
    # frame = imgtools.resize(frame, width = 300)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    faces = faceCascade.detectMultiScale(fra)
    # face_rects = fd.detect_face(gray, scale_factor, min_neighbours, minSize = (30, 30))
#    frame_clone = frame.copy()

#    for (x, y, w, h) in faces:
#        cv2.rectangle(frame_clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

#    cv2.imshow('Face', frame_clone)
#    raw_capture.truncate(0)

#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
