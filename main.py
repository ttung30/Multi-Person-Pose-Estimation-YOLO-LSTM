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
# Model
model = tf.keras.models.load_model("model.h5")
#load custom model Hieu và Tuan  
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/tung/Documents/best.pt')	
n_frame = 10
label = "  "
mp_drawing = mp.solutions.drawing_utils
mp_pose =mp.solutions.pose
pose = mp_pose.Pose()

def make_landmark(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def detect(model, lm_lis):
    global label
    # biến nhãn toàn cục
    lm_lis = np.array(lm_lis)
    #biến đổi list thành mảng numpy 
    lm_lis = np.expand_dims(lm_lis, axis=0)
    # sau khi xử lý xong list model predict
    results = model.predict(lm_lis)  
    #results  liên quan đến soft max
    print(results)
    if results[0][1] == max(results[0]):
        label = "giang tay"
    if results[0][0] == max(results[0]):
        label = "ha tay"
    #trả về nhãn # trong từng frame được crop ra  nếu như được predict biến label cục bộ sẽ được thay đổi và in lên trên ảnh đã crop 
    return label



cap = cv2.VideoCapture(0)
label2 = "Tuan"
label1 = "Hieu"
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
    thickness = 1
    for (xmin, ymin, xmax,   ymax,  confidence,  clas) in result.xyxy[0].tolist():
      with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        results = pose.process(image[int(ymin)+MARGIN:int(ymax)+MARGIN,int(xmin)+MARGIN:int(xmax)+MARGIN:])
        if results.pose_landmarks:
            c_lm = make_landmark(results)
            org = (int(xmin),int(ymin))
            c2 = (int(xmax),int(ymax))
            c1 = (int(xmin),int(ymin)) 
            cv2.rectangle(image, c1, (int(((xmax+xmin)/2)+xmax-xmin),int(((ymax+ymin)/2)+ymax-ymin)),color,4)
            if clas == 0:#nếu id  clas trong cái ảnh đó print(str(len(lm_list2))+"tuan")là Hiếu thì sẽ append vào list của Hiếu
                lm_list1.append(c_lm)
                if len(lm_list1)== n_frame:
                    #nếu số frame của tuấn đủ 10 thì detect, sau đó reset lại list Tuấ
                    detect(model, lm_list1)
                    label1 = label
                    lm_list1= []
                image = cv2.putText(image,label1, org, font, fontScale, color, thickness, cv2.LINE_AA)
            elif clas == 1:#nếu  id clas trong cái ảnh đó là Tuan  thì sẽ append vào list của Hiếu
                lm_list2.append(c_lm)
                #nếu số frame của tuấn đủ 10 thì detect, sau đó reset lại list Tuấn
                if len(lm_list2)== n_frame :
                    detect(model, lm_list2)
                    label2 = label
                    lm_list2= []
                image = cv2.putText(image,label2, org, font, fontScale, color, thickness, cv2.LINE_AA)
            
            mp_drawing.draw_landmarks(image[int(ymin)+MARGIN:int(ymax)+MARGIN,int(xmin)+MARGIN:int(xmax)+MARGIN:], results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                  )
        cv2.imshow('NguyenThanhTung ',image)
        cv2.waitKey(1)

   


