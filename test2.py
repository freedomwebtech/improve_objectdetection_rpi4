import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
import time
import pytesseract

model = YOLO('yolov8n.pt')


cap = cv2.VideoCapture('b.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
cy1=305
offset=6

while True:
    ret, frame = cap.read()
    frame=cv2.resize(frame,(1020,600))
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    results = model.predict(frame,imgsz=240)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        
        cx=int(x1+x2)//2
        cy=int(y1+y2)//2
        
#        cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
       
        cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

  
    
    cv2.imshow("FRAME", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()