# Multi-Person-Pose-Estimation-YOLO-LSTM
# To set up environment
pip install -r requirement.txt
# Model summary
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 10, 50)            36600     
                                                                 
 dropout (Dropout)           (None, 10, 50)            0         
                                                                 
 lstm_1 (LSTM)               (None, 10, 50)            20200     
                                                                 
 dropout_1 (Dropout)         (None, 10, 50)            0         
                                                                 
 lstm_2 (LSTM)               (None, 10, 50)            20200     
                                                                 
 dropout_2 (Dropout)         (None, 10, 50)            0         
                                                                 
 lstm_3 (LSTM)               (None, 50)                20200     
                                                                 
 dropout_3 (Dropout)         (None, 50)                0         
                                                                 
 dense (Dense)               (None, 3)                 153       
                                                                 
=================================================================
Total params: 97,353
Trainable params: 97,353
Non-trainable params: 0

_________________________________________________________________
#Result
![image](https://user-images.githubusercontent.com/113814417/218502630-83956d10-4394-41d4-95cb-bb48f1bb5745.png)
