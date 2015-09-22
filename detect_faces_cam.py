from detectors.face import FaceDetector
from imgtools import imgtools
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--face', required = True, help = 'path to face cascade')
args = vars(ap.parse_args())

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 64
raw_capture = PiRGBArray(camera, size=(640, 480))

fd = FaceDetector(args['face'])
time.sleep(0.1)

scale_factor = input('Enter scale factor: ')
min_neighbours = input('Enter minimum neighbours: ')

for f in camera.capture_continuous(raw_capture, format='bgr', use_video_port = True):
    frame = f.array

    frame = imgtools.resize(frame, width = 300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_rects = fd.detect_face(gray, scale_factor, min_neighbours, minSize = (30, 30))
    frame_clone = frame.copy()

    for (x, y, w, h) in face_rects:
        cv2.rectangle(frame_clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Face', frame_clone)
    raw_capture.truncate(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
