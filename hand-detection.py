import skimage.io as io
import numpy as np
import cv2
import skimage.io as io
from skimage.transform import (hough_line, hough_line_peaks,probabilistic_hough_line)
from matplotlib import cm
import matplotlib.pyplot as plt
import imutils
from skimage.feature import canny

def facedetect(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(faces)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x+20,y+10),(x+w,y+h+70),(255,255,255),-1)
    return img

def skinmask (frame):

    lower = np.array([0, 133, 77])
    upper = np.array([255, 173, 127])
    frame = imutils.resize(frame, width=400)
    converted = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
    skinMask = cv2.inRange(converted, lower, upper)

    # apply erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.dilate(skinMask, kernel, iterations=1)
    skinMask = cv2.erode(skinMask, kernel, iterations=1)
    skinMask = cv2.erode(skinMask, kernel, iterations=1)
    skinMask = cv2.dilate(skinMask, kernel, iterations=0)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)
    return np.hstack([frame, skin])

def getfingers(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    height = img.shape[0]
    width = img.shape[1]
    img = img[30:int(height) - 30, 10:int(width) - 30]
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)
    converted = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    (h,w) = np.shape(converted)
    horizontal = np.zeros(h)
    for i in range(1,h):
        horizontal[i] = np.sum(converted[i,:])
    horizontal[horizontal<0.26*np.max(horizontal)]=0
    img[horizontal>0,:,:]=0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, img) = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ret, labels = cv2.connectedComponents(img)
    print(ret)
    return img,ret

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0
count = 0
while True:
    #read frame
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    #read entered key
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        frame = facedetect(frame)
        cv2.imshow('asd', frame)
        cv2.waitKey(0)
        frame = skinmask(frame)
        cv2.imshow("images", frame)
        cv2.waitKey(0)
        frame = cv2.dilate(frame, kernel=np.ones((3, 3)), iterations=2)
        height = frame.shape[0]
        width = frame.shape[1]
        frame = frame[0:int(height), int(width / 2) + 5:int(width)]
        cv2.imshow("images", frame)
        cv2.waitKey(0)
        frame,c = getfingers(frame)
        count += c-1
        cv2.imshow("images", frame)
        cv2.waitKey(0)
        print(count)
cam.release()
cv2.destroyAllWindows()
