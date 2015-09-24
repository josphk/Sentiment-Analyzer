import cv2, os
import numpy as np
from PIL import Image
from sklearn.svm import LinearSVC

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
path = 'yalefaces'
image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.happy') or f.endswith('.sad') or f.endswith('.normal')]

test = 'yalefaces_test'
image_path_test = [os.path.join(test, f) for f in os.listdir(test) if f.endswith('.sad')]

exp = dict(sad = 0, happy = 1, normal = 0)
images = []
labels = []

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None;
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (int(w * r), height)

    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

for image_path in image_paths:
    image_pil = Image.open(image_path).convert('L')
    image = np.array(image_pil)
    nbr = exp[os.path.split(image_path)[1].split(".")[1]]
    faces = faceCascade.detectMultiScale(image)
    for (x, y, w, h) in faces:
        images.append(image[y: y+150, x: x+150].flatten())
        labels.append(nbr)

clf = LinearSVC()
clf.fit(images, labels)

for image_test in image_path_test:
    image_pil = Image.open(image_test).convert('L')
    image = np.array(image_pil)
    faces = faceCascade.detectMultiScale(image)
    for (x, y, w, h) in faces:
        print(clf.predict(image[y: y+150, x: x+150].flatten()))
