import cv2
import numpy as np
import math
import time
import os

from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)  # Webcam

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

folder = "D:\\code\\SignLang\\Dataset\\Please"

# Create folder if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture frame from webcam.")
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

        if imgCrop.size > 0:
            aspect_ratio = h / w

            if aspect_ratio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResized = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResized

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResized = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResized

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Saved image {counter}")

    elif key == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
