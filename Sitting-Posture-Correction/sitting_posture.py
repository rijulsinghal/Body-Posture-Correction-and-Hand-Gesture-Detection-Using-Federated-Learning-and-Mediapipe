import numpy as np
import mediapipe as mp
import math as m
import cv2
from gtts import gTTS
import os
from playsound import playsound  
import pyttsx3
# Initialize frame counters.
good_frames = 0
bad_frames  = 0
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)
black = (0,0,0)
font = cv2.FONT_HERSHEY_SIMPLEX

def findAngle(x1, y1, x2, y2):
    theta = m.acos( (y2 -y1)*(-y1) / (m.sqrt(
        (x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
    degree = int(180/m.pi)*theta
    return degree


def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist


    
    
mp_holisitic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))

with mp_holisitic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        h, w = frame.shape[:2]
        #Adding Border to the frame
        frame = cv2.copyMakeBorder(frame,30,30,0,0,cv2.BORDER_CONSTANT,black)

        results = holistic.process(frame)
        if(results.pose_landmarks is None):
            continue

        right_shoulder_y = results.pose_landmarks.landmark[11].y*h
        right_shoulder_z = results.pose_landmarks.landmark[11].z*w
        
        left_shoulder_y = results.pose_landmarks.landmark[12].y*h
        left_shoulder_z = results.pose_landmarks.landmark[12].z*w

        right_ear_y = results.pose_landmarks.landmark[7].y*h
        right_ear_z = results.pose_landmarks.landmark[7].z*w

        left_ear_y = results.pose_landmarks.landmark[8].y*h
        left_ear_z = results.pose_landmarks.landmark[8].z*w

        min_dis = m.degrees(m.atan(abs(left_shoulder_z-left_ear_z)/abs(left_ear_y-left_shoulder_y)))

        print(min_dis)
        if (min_dis <= 60):
            bad_frames = 0
            good_frames += 1
        else:
            good_frames = 0
            bad_frames += 1
        
        good_time = (1 / fps) * good_frames
        bad_time =  (1 / fps) * bad_frames
        if good_time > 0:
            time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
            cv2.putText(frame, time_string_good, (10, h - 20), font, 0.9, green, 2)
        else:
            time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
            cv2.putText(frame, time_string_bad, (10, h - 20), font, 0.9, red, 2)

        if bad_time > 10:
            cv2.putText(frame, "Bad Posture for more than 10 seconds", (10,20), font, 0.9, red, 2)
            bad_time = 0
            


        mp_drawing.draw_landmarks(frame,results.pose_landmarks,mp_holisitic.POSE_CONNECTIONS)
        cv2.imshow("Sitting Posture" , frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()