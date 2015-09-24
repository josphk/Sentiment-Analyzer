from detectors.face import FaceDetector
from imgtools import imgtools
from picamera.array import PiRGBArray, PiArrayOutput
import picamera
import argparse
import time
import cv2
import io
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--face', required = True, help = 'path to face cascade')
args = vars(ap.parse_args())

camera = picamera.PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
raw_capture = PiRGBArray(camera, size=(640, 480))
raw_array = PiArrayOutput(camera)

fd = FaceDetector(args['face'])
time.sleep(0.1)

scale_factor = input('Enter scale factor: ')
min_neighbours = input('Enter minimum neighbours: ')

stream = io.BytesIO()
for f in camera.capture_continuous(stream, format='jpeg'):
     stream.seek(0)
     image = Image.open(stream).convert('L')
     arr = np.array(image)
     print arr
     stream.seek(0)
     stream.truncate()

#for f in camera.capture_continuous(stream, format='gif', use_video_port = True):
#    camera.capture(stream, format='gif')
#    stream.seek(0)
#    image = Image.open(stream)
#    print(image)

    #frame = f.array

    #frame = imgtools.resize(frame, width = 300)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #face_rects = fd.detect_face(gray, scale_factor, min_neighbours, minSize = (30, 30))
    #frame_clone = frame.copy()

    #for (x, y, w, h) in face_rects:
    #    cv2.rectangle(frame_clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #cv2.imshow('Face', frame_clone)
    #raw_capture.truncate(0)

    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
