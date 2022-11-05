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

cap = cv2.VideoCapture(0)
model = keras.models.load_model("models/posture-classification")
img_size = 224
x_val = []

while cap.isOpened():

    ret, frame = cap.read()
    cv2.imshow("Yoga Correction" , frame)
    x_val = cv2.resize(frame, (img_size, img_size)) # Reshaping images to preferred size
    images_list = []
    images_list.append(np.array(x_val))
    x = np.asarray(images_list)
    result = model.predict(x)
    print(result)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()