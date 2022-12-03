import mediapipe as mp
import math as m
import cv2
import csv
import copy
import itertools
from collections import Counter
from collections import deque
import glob
import numpy as np
import os
import sys

time_string_good = None
time_string_bad = None

# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX
 
# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

mp_pose = mp.solutions.pose

def loadimages(path):
    images = []
    for img in glob.glob(path):
        images.append(img)
    return images

def getlandmarks(images):
    landmarks_matrix = []
    for img in images:
        landmarks = medpipe(img)
        if(len(landmarks) == 0):
            continue
        temp1 = []
        for i in range(0,len(landmarks)):
            temp1.append(landmarks[i].x)
            temp1.append(landmarks[i].y)
        
        landmarks_matrix.append(temp1)

    return landmarks_matrix

def medpipe(frame):
    ## Setup mediapipe instance
    #  A confidence score threshold is chosen to filter out false positives and ensure
    #  that a predicted bounding box has a certain minimum score

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image = cv2.imread(frame,0)
        height, width = image.shape[:2] #getting the shape of the image.
        # Recolor image to RGB
        #cv2.cvtColor() method is used to convert an image from one color space to another.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection and handle error conditions if any
        results = pose.process(image)
        if results.pose_landmarks is None:
            return []
        landmarks = results.pose_landmarks.landmark
        return landmarks       

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    # temp_landmark_list = list(
    #     itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    # max_value = max(list(map(abs, temp_landmark_list)))

    # def normalize_(n):
    #     return n / max_value

    # temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list
    

labels = os.listdir("Dataset/Yoga-Dataset/TRAIN")



# images = loadimages('Dataset/Yoga-Dataset/TRAIN/ardha pincha mayurasana/*')
# images1 = loadimages('Dataset/Yoga-Dataset/TRAIN/goddess/*')
# images2 = loadimages('Dataset/Yoga-Dataset/TRAIN/bitilasana/*')
# images3 = loadimages('Dataset/Yoga-Dataset/TRAIN/tree/*')
# images4 = loadimages('Dataset/Yoga-Dataset/TRAIN/warrior2/*')
# images5 = loadimages('Dataset/Yoga-Dataset/TRAIN/chaturanga dandasana/*')
# images6 = loadimages('Dataset/Yoga-Dataset/TRAIN/hanumanasana/*')
# images7 = loadimages('Dataset/Yoga-Dataset/TRAIN/krounchasana/*')
# images8 = loadimages('Dataset/Yoga-Dataset/TRAIN/matsyasana/*')
# images9 = loadimages('Dataset/Yoga-Dataset/TRAIN/paripurna navasana/*')
images10 = loadimages('Dataset/Yoga-Dataset/TRAIN/parivrtta trikonasana/*')
images11 = loadimages('Dataset/Yoga-Dataset/TRAIN/purvottanasana/*')
images12 = loadimages('Dataset/Yoga-Dataset/TRAIN/dandasana/*')


csv_path = 'YogaDetection-Medipipe/yoga_keypoint.csv'

# landmark_list = getlandmarks(images)

# landmark_list = pre_process_landmark(landmark_list)
# # print(landmark_list)
# for i in landmark_list:
#     with open(csv_path, 'a', newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([0, *i])


# landmark_list = getlandmarks(images1)
# landmark_list = pre_process_landmark(landmark_list)
# # print(landmark_list)
# for i in landmark_list:
#     with open(csv_path, 'a', newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([1, *i])

# landmark_list = getlandmarks(images2)
# landmark_list = pre_process_landmark(landmark_list)

# for i in landmark_list:
#     with open(csv_path, 'a', newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([2, *i])

# landmark_list = getlandmarks(images3)
# landmark_list = pre_process_landmark(landmark_list)

# for i in landmark_list:
#     with open(csv_path, 'a', newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([3, *i])

# landmark_list = getlandmarks(images4)
# landmark_list = pre_process_landmark(landmark_list)

# for i in landmark_list:
#     with open(csv_path, 'a', newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([4, *i])
# landmark_list = getlandmarks(images5)
# landmark_list = pre_process_landmark(landmark_list)
# for i in landmark_list:
#     with open(csv_path, 'a', newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([5, *i])

# landmark_list = getlandmarks(images6)
# landmark_list = pre_process_landmark(landmark_list)

# for i in landmark_list:
#     with open(csv_path, 'a', newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([6, *i])
# landmark_list = getlandmarks(images7)
# landmark_list = pre_process_landmark(landmark_list)

# for i in landmark_list:
#     with open(csv_path, 'a', newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([7, *i])

# landmark_list = getlandmarks(images8)
# landmark_list = pre_process_landmark(landmark_list)

# for i in landmark_list:
#     with open(csv_path, 'a', newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([8, *i])

# landmark_list = getlandmarks(images9)
# landmark_list = pre_process_landmark(landmark_list)

# for i in landmark_list:
#     with open(csv_path, 'a', newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([9, *i])

landmark_list = getlandmarks(images10)
landmark_list = pre_process_landmark(landmark_list)

for i in landmark_list:
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([10, *i])

landmark_list = getlandmarks(images11)
landmark_list = pre_process_landmark(landmark_list)

for i in landmark_list:
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([11, *i])

landmark_list = getlandmarks(images12)
landmark_list = pre_process_landmark(landmark_list)

for i in landmark_list:
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([12, *i])