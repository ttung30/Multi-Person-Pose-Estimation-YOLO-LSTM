import numpy as np
import pandas as pd


import tensorflow as tf

from sklearn.model_selection import train_test_split
class LSTM_model(tf.keras.models):
    def __init__(self):
        super(LSTM_model,self).__init__()
        self.lstm1=tf.keras.layers.LSTM(units = 50, return_sequences=True)
        self.drop1=tf.keras.layers.Dropout(0.2)
        self.lstm2=tf.keras.layers.LSTM(units = 50, return_sequences=True)
        self.drop2=tf.keras.layers.Dropout(0.2)
        self.lstm3=tf.keras.layers.LSTM(units=50, return_sequences= True)
        self.drop3=tf.keras.layers.Dropout(0.2)
        self.lstm4=tf.keras.layers.LSTM(units=50)
        self.drop4=tf.keras.layers.Dropout(0.2)
        self.fc1=tf.keras.layers.Dense(1,activation="sigmoid")
    def call(self, x):
        x=self.lstm1(x)
        x=self.drop1(x)
        x=self.lstm2(x)
        x=self.drop2(x)
        x=self.lstm3(x)
        x=self.drop3(x)
        x=self.lstm4(x)
        x=self.drop4(x)
        x=self.fc1(x)

bodyswing_df = pd.read_csv("SWING.txt")
handswing_df = pd.read_csv("HANDSWING.txt")

X = []
y = []
no_of_timesteps = 10

dataset = bodyswing_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(1)

dataset = handswing_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(0)

X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model =LSTM_model()
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "binary_crossentropy")

model.fit(X_train, y_train, epochs=16, batch_size=32,validation_data=(X_test, y_test))
model.save("model.h5")

