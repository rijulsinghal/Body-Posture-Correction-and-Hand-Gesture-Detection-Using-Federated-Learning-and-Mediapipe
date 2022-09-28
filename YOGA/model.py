from turtle import left
import numpy as np
import mediapipe as mp
import math as m
import glob
import cv2

mp_pose = mp.solutions.pose


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


def dist(x1,y1,x2,y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5


def loadimages(path):
    images = []
    for img in glob.glob(path):
        images.append(img)
    return images

def findAngle(x1, y1, x2, y2):
    theta = m.acos( (y2 -y1)*(-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
    degree = int(180/m.pi)*theta
    return degree

def calculate_angle(x1,y1,x2,y2):
    angle = m.atan2(abs(y2-y1),abs(x2-x1))
    return (m.degrees(angle))

from asyncio.windows_events import NULL
from logging import exception


def find_angle_downdog(downdog):
    ans = 0
    for image in downdog:
        left_elbow_x = image[16][0]
        left_elbow_y = image[16][1]
        left_ankle_x = image[28][0]
        left_ankle_y = image[28][1]
        ans = ans + calculate_angle(left_ankle_x,left_ankle_y,left_elbow_x,left_elbow_y)

    return ans/len(downdog)

def find_angle_goddess(goddess):
    left = 0
    right = 0
    for image in goddess:
        left_hip_x = image[24][0]
        left_hip_y = image[24][1]
        right_hip_x = image[23][0]
        right_hip_y = image[23][1]
        left_heel_x = image[30][0]
        left_heel_y = image[30][1]
        right_heel_x = image[29][0]
        right_heel_y = image[29][1]
        left_knee_x = image[26][0]
        left_knee_y = image[26][1]
        right_knee_x = image[25][0]
        right_knee_y = image[25][1]
        left = left + 2*m.degrees(m.atan(dist(left_hip_x,left_hip_y,left_knee_x,left_knee_y)/dist(left_knee_x,left_knee_y,left_heel_x,left_heel_y)))
        right = right + 2*m.degrees(m.atan(dist(right_hip_x,right_hip_y,right_knee_x,right_knee_y)/dist(right_knee_x,right_knee_y,right_heel_x,right_heel_y)))

    return [left/len(goddess),right/len(goddess)]

def getlandmarks(images):
    landmarks_matrix = []
    for img in images:
        landmarks = medpipe(img)
        if(len(landmarks) == 0):
            continue
        temp1 = []
        for i in range(0,len(landmarks)):
            temp = []
            temp.append(landmarks[i].x)
            temp.append(landmarks[i].y)
            temp.append(landmarks[i].z)
            temp.append(landmarks[i].visibility)
            # print(temp)
            temp1.append(temp)
        # print(temp1)
        landmarks_matrix.append(temp1)  
    print(len(landmarks_matrix))
    return landmarks_matrix


# downdog_image = loadimages('Dataset/Yoga-Dataset/DATASET/TRAIN/downdog/*')
goddess_image = loadimages('../Dataset/Yoga-Dataset/DATASET/TRAIN/goddess/*')
# plank_image = loadimages('Dataset/Yoga-Dataset/DATASET/TRAIN/plank/*')
# tree_image = loadimages('Dataset/Yoga-Dataset/DATASET/TRAIN/tree/*')
# warrior2_image = loadimages('Dataset/Yoga-Dataset/DATASET/TRAIN/warrior2/*')


downdog = []
goddess = []
plank = []
tree = []
warrior2 = []

# downdog = getlandmarks(downdog_image)
goddess = getlandmarks(goddess_image)
# plank = getlandmarks(plank_image)
# tree = getlandmarks(tree_image)
# warrior2 = getlandmarks(warrior2_image)


# find_angle_downdog(downdog)
find_angle_goddess(goddess)
# TODO

# find_angle_plank(plank)
# find_angle_tree(tree)
# find_angle_warrior2(warrior2)