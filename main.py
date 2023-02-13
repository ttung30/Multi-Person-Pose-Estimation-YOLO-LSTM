from PIL import Image as ima
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
import cv2
import numpy as np
import tensorflow as tf
import torch

lm_list1 = []
lm_list2 = []
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
model = tf.keras.models.load_model("model.h5")
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')	
n_frame = 10
label = "  "
label2 = "Tuan"
label1 = "Hieu"
mp_drawing = mp.solutions.drawing_utils
mp_pose =mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(2)
def landmark(results):
    lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        lm.append(lm.x)
        lm.append(lm.y)
        lm.append(lm.z)
        lm.append(lm.visibility)
    return lm

def detect(model, lm_lis):
    global label
    lm_lis = np.array(lm_lis)
    lm_lis = np.expand_dims(lm_lis, axis=0)
    results = model.predict(lm_lis)  
    print(results)
    if results[0][1] == max(results[0]):
        label = "giang tay"
    if results[0][0] == max(results[0]):
        label = "ha tay"
    if results[0][2] == max(results[0]):
        label = "ngoi"
    return label


while cap.isOpened():    
    ret, frame = cap.read()
    if not ret:
      break
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False    
    result = yolo_model(image)
    image.flags.writeable = True   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    MARGIN=10
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    color1 = (0, 0 , 255)
    thickness = 1
    for (xmin, ymin, xmax,   ymax,  confidence,  clas) in result.xyxy[0].tolist():
      with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        results = pose.process(image[int(ymin)+MARGIN:int(ymax)+MARGIN,int(xmin)+MARGIN:int(xmax)+MARGIN:])
        if results.pose_landmarks:
            c_lm = landmark(results)
            org = (int(xmin),int(ymin))
            c1 = (int(xmin),int(ymin)) 
            cv2.rectangle(image, c1, (int(((xmax+xmin)/2)+xmax-xmin),int(((ymax+ymin)/2)+ymax-ymin)),color,4)
            if clas == 0:
                lm_list1.append(c_lm)
                if len(lm_list1)== n_frame:
                    
                    detect(model, lm_list1)
                    label1 = label
                    lm_list1= []
                image = cv2.putText(image,label1, org, font, fontScale, color, thickness, cv2.LINE_AA)
            elif clas == 1:
                lm_list2.append(c_lm)
          
                if len(lm_list2)== n_frame :
                    detect(model, lm_list2)
                    label2 = label
                    lm_list2= []
                image = cv2.putText(image,label2, org, font, fontScale, color1, thickness, cv2.LINE_AA)
            mp_drawing.draw_landmarks(image[int(ymin)+MARGIN:int(ymax)+MARGIN,int(xmin)+MARGIN:int(xmax)+MARGIN:], results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66)), 
                                mp_drawing.DrawingSpec(color=(245,66,230)) 
                                  )
        cv2.imshow('NguyenThanhTung ',image)
        cv2.waitKey(1)

   

