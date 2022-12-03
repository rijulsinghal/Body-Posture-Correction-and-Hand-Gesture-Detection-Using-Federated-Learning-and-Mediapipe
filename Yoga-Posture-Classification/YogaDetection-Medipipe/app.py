import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import tensorflow as tf

import cv2 as cv2
import numpy as np
import mediapipe as mp


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

class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='yoga_keypoint_classifier.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index


mp_pose = mp.solutions.pose


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)


    keypoint_classifier = KeyPointClassifier()
    
    # Read labels ###########################################################
    with open('YogaDetection-Medipipe/yoga_labels.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]


    while True:


        # Process Key (ESC: end) #################################################
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)


        ret, frame = cap.read()
        if not ret:
            break
        
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
            temp_landmark_list = copy.deepcopy(landmarks)

    # Convert to relative coordinates
            base_x, base_y = 0, 0
            for index, landmark_point in enumerate(temp_landmark_list):
                if index == 0:
                    base_x, base_y = landmark_point[0], landmark_point[1]

                temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
                temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

            index = keypoint_classifier(temp_landmark_list)

            cv2.putText(frame, keypoint_classifier_labels[index], 
                           [15,15], 
                           2,0.5,green,1
                                )
    # with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    #     image = cv.imread(frame,0)
    #     height, width = image.shape[:2] #getting the shape of the image.
    #     # Recolor image to RGB
    #     #cv2.cvtColor() method is used to convert an image from one color space to another.
    #     image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    #     image.flags.writeable = False

    #     # Make detection and handle error conditions if any
    #     results = pose.process(image)
    #     if results.pose_landmarks is None:
    #         return []
    #     landmarks = results.pose_landmarks.landmark



if __name__ == '__main__':
    main()
