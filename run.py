import cv2 
from ultralytics import YOLO
  
  
# define a video capture object 
vid = cv2.VideoCapture(0) 
model = YOLO("./runs/detect/train2/weights/best.pt")
  
ret = False
while(ret): 
      
    ret, frame = vid.read() 

    result = model.predict(frame)
    frame = result.plot()
  
    cv2.imshow('frame', frame) 
      
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
vid.release() 
cv2.destroyAllWindows() 