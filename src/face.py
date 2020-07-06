import numpy as np 
import cv2
import cv2 as cv
import pickle


 
# Haar cascade  to detect face in Image
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

# Local Binary Patterns Histogram
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

labels = {"person_name":1}
with open("pickles/face-labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    # Inverting the actual labels
    labels = {val:key for key, val in og_labels.items()}


cap = cv2.VideoCapture(0)


while(True):
    # captures Video frame-by-frame
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor =1.5, minNeighbors=5)                                                                                                                                                                                                  
    for (x, y, w, h) in faces:
   
        # Region of Interest
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = frame[y:y+h, x:x+w]
        
        id_, conf = recognizer.predict(roi_gray)
        if conf>=4 and conf<= 85:
     
            
            # put labels on image
            font =cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        
        

        # Rectangle Boundary
        color = (255,0,0)  # BGR Colorcode (0-255)
        stroke =2
        end_cord_x= x+w
        end_cord_y = y+h
        cv2.rectangle(frame,(x,y),(end_cord_x, end_cord_y), color,stroke)


    
        smile = smile_cascade.detectMultiScale(roi_gray)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color,(sx,sy), (sx+sw, sy+sh),(0,255,0),1)
    
    
    # Displaying the resulting Frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0XFF == ord('q'):
        break


# When Execution is done, release the capture
cap.release()
cv2.destoryAllWindows()