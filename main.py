import cv2
import math
import subprocess
import numpy as np
import HandTrackingModule as HTM


def set_volume(volume_level):
    applescript = f'osascript -e "set volume output volume {volume_level}"'
    subprocess.run(applescript, shell=True)


cap = cv2.VideoCapture(1)
detector = HTM.handDetector()
while True:
    isTrue, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame)
    if len(lmList) != 0:
        x4, y4 = lmList[4][1], lmList[4][2]
        x8, y8 = lmList[8][1], lmList[8][2]
        cv2.line(frame, (x4, y4), (x8, y8), (255, 0, 255), 2)
        length = math.dist([x4, y4], [x8, y8])
        vol = np.interp(length, [28, 350], [0, 100])
        vol = int(vol)
        set_volume(vol)
        cv2.putText(frame, f"Volume: {vol} %", (30, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.imshow("Image", frame)
    cv2.waitKey(1)
