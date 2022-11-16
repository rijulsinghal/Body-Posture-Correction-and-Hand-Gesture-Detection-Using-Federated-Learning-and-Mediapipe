import numpy as np
import mediapipe as mp
import math as m
import glob
import cv2
import os
import cv2
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import keras
import tensorflow
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout, BatchNormalization, Activation

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score 


green = (127, 255, 0)
cap = cv2.VideoCapture(0)
model = keras.models.load_model("Models/posture-classification")
labels = os.listdir("C:/Users/HP/Documents/Capstone-Project/Implementation/Dataset/Yoga-Dataset/TRAIN")
img_size = 224
x_val = []

def main_func():
    while cap.isOpened():

        ret, frame = cap.read()
        cv2.imshow("Yoga Correction" , frame)
        x_val = cv2.resize(frame, (img_size, img_size)) # Reshaping images to preferred size
        images_list = []
        images_list.append(np.array(x_val))
        x = np.asarray(images_list)
        result = model.predict(x)
        result = result[0]
        max_val = max(result)
        index = np.where(result == max_val)[0][0]
        cv2.putText(frame, labels[index] , (5,25) ,2, 0.5 ,green , 1)
        # print(labels[index])
        ret, buffer = cv2.imencode('.jpg', frame)
        image = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')  # concat frame one by one and show result
                
    cap.release()
    cv2.destroyAllWindows()