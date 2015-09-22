import cv2

class FaceDetector:
  def __init__(self, faceCascadePath):
    self.faceCascade = cv2.CascadeClassifier(faceCascadePath)

  def detect_face(self, image, scaleFactor, minNeighbours, minSize):
    rects = self.faceCascade.detectMultiScale(image, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    return rects
