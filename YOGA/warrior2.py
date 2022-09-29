import numpy as np
import mediapipe as mp
import math as m
import cv2

red = (50, 50, 255)
green = (127, 255, 0)

def calculate_angle(x1, y1, x2, y2, x3, y3):
    angle = m.degrees(m.atan2(y3 - y2, x3 - x2) - m.atan2(y1 - y2, x1 - x2))
    if (angle < 0):
        angle += 360
    return angle

mp_holisitic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
with mp_holisitic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:

    while cap.isOpened():
        # Start capturing frame

        ret, frame = cap.read()
        results = holistic.process(frame)
    
        if(results.pose_landmarks is None):
            continue

        # Obtaining Landmark Coordinates for Warrior II
    
        left_shoulder_x = results.pose_landmarks.landmark[11].x
        left_shoulder_y = results.pose_landmarks.landmark[11].y
        right_shoulder_x = results.pose_landmarks.landmark[12].x
        right_shoulder_y = results.pose_landmarks.landmark[12].y

        left_elbow_x = results.pose_landmarks.landmark[13].x
        left_elbow_y = results.pose_landmarks.landmark[13].y
        right_elbow_x = results.pose_landmarks.landmark[14].x
        right_elbow_y = results.pose_landmarks.landmark[14].y

        left_wrist_x = results.pose_landmarks.landmark[15].x
        left_wrist_y = results.pose_landmarks.landmark[15].y
        right_wrist_x = results.pose_landmarks.landmark[16].x
        right_wrist_y = results.pose_landmarks.landmark[16].y

        left_hip_x = results.pose_landmarks.landmark[23].x
        left_hip_y = results.pose_landmarks.landmark[23].y
        right_hip_x = results.pose_landmarks.landmark[24].x
        right_hip_y = results.pose_landmarks.landmark[24].y

        left_knee_x = results.pose_landmarks.landmark[25].x
        left_knee_y = results.pose_landmarks.landmark[25].y
        right_knee_x = results.pose_landmarks.landmark[26].x
        right_knee_y = results.pose_landmarks.landmark[26].y

        left_ankle_x = results.pose_landmarks.landmark[27].x
        left_ankle_y = results.pose_landmarks.landmark[27].y
        right_ankle_x = results.pose_landmarks.landmark[28].x
        right_ankle_y = results.pose_landmarks.landmark[28].y

        # Calculating Angles

        left_elbow_angle = calculate_angle(left_shoulder_x, left_shoulder_y, left_elbow_x, left_elbow_y, left_wrist_x, left_wrist_y)
        right_elbow_angle = calculate_angle(left_shoulder_x, left_shoulder_y, left_elbow_x, left_elbow_y, left_wrist_x, left_wrist_y)
        left_shoulder_angle = calculate_angle(left_elbow_x, left_elbow_y, left_shoulder_x, left_shoulder_y, left_hip_x, left_hip_y)
        right_shoulder_angle = calculate_angle(left_elbow_x, left_elbow_y, left_shoulder_x, left_shoulder_y, left_hip_x, left_hip_y)
        left_knee_angle = calculate_angle(left_hip_x, left_hip_y, left_knee_x, left_knee_y, left_ankle_x, left_ankle_y)
        right_knee_angle = calculate_angle(right_hip_x, right_hip_y, right_knee_x, right_knee_y, right_ankle_x, right_ankle_y)

        # Check if Warrior II Conditions are met

        # # Check if Left Elbow is Straight
        if (left_elbow_angle > 165) and (left_elbow_angle < 195):
            cv2.putText(frame, str(left_elbow_angle), 
                           tuple(np.multiply([left_elbow_x, left_elbow_y], [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, green, 3, cv2.LINE_AA
                                )
        else:
            cv2.putText(frame, str(left_elbow_angle), 
                           tuple(np.multiply([left_elbow_x, left_elbow_y], [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, red, 3, cv2.LINE_AA
                                )
        
        # # Check if Right Elbow is Straight
        if (right_elbow_angle > 165) and (right_elbow_angle < 195):
            cv2.putText(frame, str(right_elbow_angle), 
                           tuple(np.multiply([right_elbow_x, right_elbow_y], [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, green, 3, cv2.LINE_AA
                                )
        else:
            cv2.putText(frame, str(right_elbow_angle), 
                           tuple(np.multiply([right_elbow_x, right_elbow_y], [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, red, 3, cv2.LINE_AA
                                )

        # # Check if Left and Right Shoulders are in Correct Position
        if (left_shoulder_angle > 80) and (left_shoulder_angle < 110):
            cv2.putText(frame, str(left_shoulder_angle), 
                           tuple(np.multiply([left_shoulder_x, left_shoulder_y], [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, green, 3, cv2.LINE_AA
                                )
        else:
            cv2.putText(frame, str(left_shoulder_angle), 
                           tuple(np.multiply([left_shoulder_x, left_shoulder_y], [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, red, 3, cv2.LINE_AA
                                )
        
        # # Check if Right Shoulder is in Correct Position
        if (right_shoulder_angle > 80) and (right_shoulder_angle < 110):
            cv2.putText(frame, str(right_shoulder_angle), 
                           tuple(np.multiply([right_shoulder_x, right_shoulder_y], [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, green, 3, cv2.LINE_AA
                                )
        else:
            cv2.putText(frame, str(right_shoulder_angle), 
                           tuple(np.multiply([right_shoulder_x, right_shoulder_y], [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, red, 3, cv2.LINE_AA
                                )
       
        # # Check if one Knee is Straight and the other Knee is Bent
        if ((left_knee_angle > 165) and (left_knee_angle < 195)) and ((right_knee_angle > 90) and (right_knee_angle < 120)) or ((left_knee_angle > 90) and (left_knee_angle < 120)) or ((right_knee_angle > 165) and (right_knee_angle < 195)):
            cv2.putText(frame, str(left_knee_angle), 
                           tuple(np.multiply([left_knee_x, left_knee_y], [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, green, 3, cv2.LINE_AA
                                )
            cv2.putText(frame, str(right_knee_angle), 
                           tuple(np.multiply([right_knee_x, right_knee_y], [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, green, 3, cv2.LINE_AA
                                )

        else:
            cv2.putText(frame, str(left_knee_angle), 
                           tuple(np.multiply([left_knee_x, left_knee_y], [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, red, 3, cv2.LINE_AA
                                )
            cv2.putText(frame, str(right_knee_angle), 
                           tuple(np.multiply([right_knee_x, right_knee_y], [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, red, 3, cv2.LINE_AA
                                )

        # Output
        mp_drawing.draw_landmarks(frame,results.pose_landmarks,mp_holisitic.POSE_CONNECTIONS)
        cv2.imshow("Warrior II Pose" , frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()