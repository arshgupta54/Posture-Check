import cv2 
import mediapipe as mp 
from mediapipe.tasks.python import vision 
from mediapipe.tasks.python import text 
from mediapipe.tasks.python import audio 

camera = cv2.VideoCapture(0) 

while camera.isOpened(): 
    ret, frame = camera.read() 
    if not ret: 
        break 
    
    cv2.imshow("poop", frame) 

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 

camera.release() 
cv2.destroyAllWindows()