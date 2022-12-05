import numpy as np
import mediapipe as mp
import math as m
import cv2

red = (50, 50, 255)
green = (127, 255, 0)
black = (0,0,0)

def findAngle(x1, y1, x2, y2):
    theta = m.acos( (y2 -y1)*(-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
    degree = int(180/m.pi)*theta
    return degree

def main():
    mp_holisitic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(1)
    with mp_holisitic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            results = holistic.process(frame)
            if(results.pose_landmarks is None):
                continue
            left_elbow_x = results.pose_landmarks.landmark[20].x
            left_elbow_y = results.pose_landmarks.landmark[20].y
            left_ankle_x = results.pose_landmarks.landmark[28].x
            left_ankle_y = results.pose_landmarks.landmark[28].y
            left_hip_x = results.pose_landmarks.landmark[23].x
            left_hip_y = results.pose_landmarks.landmark[23].y
            ans = findAngle(left_ankle_x,left_ankle_y,left_elbow_x,left_elbow_y)
            print(ans)

            #Adding Border to the frame
            frame = cv2.copyMakeBorder(frame,30,30,0,0,cv2.BORDER_CONSTANT,black)

            # # PRINTING ANGLE AND ERROR DETECTION
            if ans >= 85 and ans <= 105:
                cv2.putText(frame,'Angle: ' + str(round(ans,1)) , (5,25) ,2, 0.5 ,green , 1)
            else:
                cv2.putText(frame,'Angle: ' + str(round(ans,1)) , (5,25) ,2, 0.5 ,red , 1)
                


            mp_drawing.draw_landmarks(frame,results.pose_landmarks,mp_holisitic.POSE_CONNECTIONS)
            # cv2.imshow("Downdog Posture" , frame)
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break

            ret, buffer = cv2.imencode('.jpg', frame)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')  # concat frame one by one and show result
    cap.release()
    cv2.destroyAllWindows()