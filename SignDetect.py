import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("D:\code\SignLang\model/keras_model.h5" , "D:\code\SignLang\model/labels.txt")
offset = 20
imgSize = 300
# counter = 0

labels = ["Hello","Thankyou","Yes", "No", "Good", "Bad", "Goodluck", "Iloveyou", "Livelong", "Peace", "Please", "Okay", "Callme"]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture frame from webcam.")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        
        if imgCrop.size > 0:  # Check if imgCrop is not empty
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgWhite[:, wGap: wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgWhite[hGap: hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x-offset+400, y-offset+60-50), (0,255,0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y+h + offset), (0,255,0), 4)

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
