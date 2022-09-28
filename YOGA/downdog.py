import numpy as np
import mediapipe as mp
import math as m
import cv2

red = (50, 50, 255)
green = (127, 255, 0)

def findAngle(x1, y1, x2, y2):
    theta = m.acos( (y2 -y1)*(-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
    degree = int(180/m.pi)*theta
    return degree


mp_holisitic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
with mp_holisitic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        results = holistic.process(frame)
        if(results.pose_landmarks is None):
            continue
        left_elbow_x = results.pose_landmarks.landmark[16].x
        left_elbow_y = results.pose_landmarks.landmark[16].y
        left_ankle_x = results.pose_landmarks.landmark[28].x
        left_ankle_y = results.pose_landmarks.landmark[28].y
        left_hip_x = results.pose_landmarks.landmark[23].x
        left_hip_y = results.pose_landmarks.landmark[23].y
        ans = findAngle(left_ankle_x,left_ankle_y,left_elbow_x,left_elbow_y)
        print(ans)
        
        # # PRINTING ANGLE AND ERROR DETECTION
        if ans >= 85 and ans <= 105:
            cv2.putText(frame, str(ans), 
                           tuple(np.multiply([left_hip_x,left_hip_y], [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, green, 3, cv2.LINE_AA
                                )
        else:
            cv2.putText(frame, str(ans), 
                           tuple(np.multiply([left_hip_x,left_hip_y], [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, red, 3, cv2.LINE_AA
                                )
       


        mp_drawing.draw_landmarks(frame,results.pose_landmarks,mp_holisitic.POSE_CONNECTIONS)
        cv2.imshow("Downdog Posture" , frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()