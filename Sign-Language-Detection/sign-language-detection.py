from logging import exception
import numpy as np
import mediapipe as mp
import math as m
import glob
import cv2
import re

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

mydict = {'0': [0,0]}

def loadimages(path):
    images = []
    for img in glob.glob(path):
        images.append(img)
    return images


def medpipe(frame,ch):

    with mp_hands.Hands(static_image_mode=True, max_num_hands = 2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        
        image = cv2.imread(frame,0)
        height, width = image.shape[:2] #getting the shape of the image.
        # Recolor image to RGB
        # cv2.cvtColor() method is used to convert an image from one color space to another.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = hands.process(image)
        # print(results.multi_hand_landmarks)
        # return results.multi_hand_landmarks
        hands = []

        if results.multi_hand_landmarks:
            for hand_no,hand_landmarks in enumerate(results.multi_hand_landmarks):

                # print(f'HAND NUMBER: {hand_no+1}')
                # print('-----------------------')
                hand = []
                for i in range(21):
                    temp = []
                    temp.append(hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x)
                    temp.append(hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y) 
                    temp.append(hand_landmarks.landmark[mp_hands.HandLandmark(i).value].z)
                    hand.append(temp)
                hands.append(hand)

        if(len(hands) == 1):
            hands.append([])
            mydict[ch][0] = mydict[ch][0]+1
        elif len(hands) == 2:
            mydict[ch][1] = mydict[ch][1]+1

        # print(len(hands))
        return hands


def getlandmarks(images,ch):
    if(mydict.get(ch) is None):
        mydict[ch] = [0,0]
    landmarks_matrix = []
    for img in images:
        landmarks = medpipe(img,ch)
        if(len(landmarks) == 2):
            landmarks_matrix.append(landmarks)

    return landmarks_matrix

image_1 = loadimages('Dataset/Sign-Language-Dataset/1/*')
image_2 = loadimages('Dataset/Sign-Language-Dataset/2/*')
image_3 = loadimages('Dataset/Sign-Language-Dataset/3/*')
image_4 = loadimages('Dataset/Sign-Language-Dataset/4/*')
image_5 = loadimages('Dataset/Sign-Language-Dataset/5/*')
image_6 = loadimages('Dataset/Sign-Language-Dataset/6/*')
image_7 = loadimages('Dataset/Sign-Language-Dataset/7/*')
image_8 = loadimages('Dataset/Sign-Language-Dataset/8/*')
image_9 = loadimages('Dataset/Sign-Language-Dataset/9/*')
image_A = loadimages('Dataset/Sign-Language-Dataset/A/*')
image_B = loadimages('Dataset/Sign-Language-Dataset/B/*')
image_C = loadimages('Dataset/Sign-Language-Dataset/C/*')
image_D = loadimages('Dataset/Sign-Language-Dataset/D/*')
image_E = loadimages('Dataset/Sign-Language-Dataset/E/*')
image_F = loadimages('Dataset/Sign-Language-Dataset/F/*')
image_G = loadimages('Dataset/Sign-Language-Dataset/G/*')
image_H = loadimages('Dataset/Sign-Language-Dataset/H/*')
image_I = loadimages('Dataset/Sign-Language-Dataset/I/*')
image_J = loadimages('Dataset/Sign-Language-Dataset/J/*')
image_K = loadimages('Dataset/Sign-Language-Dataset/K/*')
image_L = loadimages('Dataset/Sign-Language-Dataset/L/*')
image_M = loadimages('Dataset/Sign-Language-Dataset/M/*')
image_N = loadimages('Dataset/Sign-Language-Dataset/N/*')
image_O = loadimages('Dataset/Sign-Language-Dataset/O/*')
image_P = loadimages('Dataset/Sign-Language-Dataset/P/*')
image_Q = loadimages('Dataset/Sign-Language-Dataset/Q/*')
image_R = loadimages('Dataset/Sign-Language-Dataset/R/*')
image_S = loadimages('Dataset/Sign-Language-Dataset/S/*')
image_T = loadimages('Dataset/Sign-Language-Dataset/T/*')
image_U = loadimages('Dataset/Sign-Language-Dataset/U/*')
image_V = loadimages('Dataset/Sign-Language-Dataset/V/*')
image_W = loadimages('Dataset/Sign-Language-Dataset/W/*')
image_X = loadimages('Dataset/Sign-Language-Dataset/X/*')
image_Y = loadimages('Dataset/Sign-Language-Dataset/Y/*')
image_Z = loadimages('Dataset/Sign-Language-Dataset/Z/*')



res = getlandmarks(image_C,'C')
print(len(res))
print(res[0])
print(mydict)