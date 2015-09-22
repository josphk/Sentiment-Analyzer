from __future__ import print_function
from ../detector.face import FaceDetector
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--face', required = True, help = 'path to face cascade')
ap.add_argument('i', '--image', required = True, help = 'path to image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

fd = FaceDetector(args['face'])
scale_factor = input('Enter scale factor: ')
min_neighbours = input('Enter minimum neighbours: ')

face_rects = fd.detect(gray, scale_factor, min_neighbours)
print('{} face(s) detected'.format(len(face_rects)))

for (x,y,w,h) in face_rects:
  cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Faces', image)
cv2.waitKey(0)
