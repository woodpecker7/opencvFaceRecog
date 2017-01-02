import cv2
import numpy as np
import time
cap=cv2.VideoCapture(0)
s=1
path="haarcascade_frontalface_default.xml"
cameraCapture = cv2.VideoCapture(0)
fps = 30 # an assumption
size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter('MyOutputVid.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)
numFramesRemaining = 5 * fps - 1
while True:
    ret,frame=cap.read()
    cap.set(4,480)
#    cv2.imshow("me",frame)
    color=(0,0,255)
    classfier=cv2.CascadeClassifier(path)
    size=frame.shape[:2]
    image=np.zeros(size,dtype=np.float16)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(image, image)
    divisor=8
    h, w = size
    minSize=(w/divisor, h/divisor)
    faceRects = classfier.detectMultiScale(image, 1.2, 2, cv2.CASCADE_SCALE_IMAGE,minSize)
    if len(faceRects)>0:
        for faceRect in faceRects: 
            x, y, w, h = faceRect
            cv2.rectangle(frame, (x, y), (x+w, y+h), color)
    cv2.imshow("test", frame)
    videoWriter.write(frame)
    numFramesRemaining -= 1
    if numFramesRemaining<=0:
        break
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap.release()
videoWriter.release()
cv2.destroyAllWindows()