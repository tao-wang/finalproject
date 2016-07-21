import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    img = cv2.resize(img, None, fx=1/3.0, fy=1/3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # do edge detection on faces
    for (x,y,w,h) in faces: 
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        roi_edges = cv2.Canny(roi_color, 50, 200)
        roi_edges[roi_edges==255] = 128
        roi_edges[roi_edges==0] = 255
        roi_edges[roi_edges==128] = 0
        to_insert = np.zeros(roi_color.shape)
        for channel in range(3):
        	to_insert[:,:,channel] = roi_edges
        img[y:y+h, x:x+w] = to_insert
        # eyes = eye_cascade.detectMultiScale(roi_gray, minSize=(50,50))
        # for (ex,ey,ew,eh) in eyes[:2]:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('webcam', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()