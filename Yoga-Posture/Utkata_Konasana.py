import numpy as np
import mediapipe as mp
import math as m
import cv2
import math

red = (50, 50, 255)
green = (127, 255, 0)
black = (0, 0, 0)


def dist(x1,y1,x2,y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5

def calculate_angle(x1,y1,x2,y2):
    angle = math.atan2(abs(y2-y1),abs(x2-x1))
    return (math.degrees(angle))

def main():
    mp_holisitic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    with mp_holisitic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            results = holistic.process(frame)
            

            if(results.pose_landmarks is None):
                continue


            left_knee_x = results.pose_landmarks.landmark[26].x
            left_knee_y = results.pose_landmarks.landmark[26].y
            right_knee_x = results.pose_landmarks.landmark[25].x
            right_knee_y = results.pose_landmarks.landmark[25].y
            left_heel_x = results.pose_landmarks.landmark[28].x
            left_heel_y = results.pose_landmarks.landmark[28].y
            right_heel_x = results.pose_landmarks.landmark[27].x
            right_heel_y = results.pose_landmarks.landmark[27].y
            left_hip_x = results.pose_landmarks.landmark[24].x
            left_hip_y = results.pose_landmarks.landmark[24].y
            right_hip_x = results.pose_landmarks.landmark[23].x
            right_hip_y = results.pose_landmarks.landmark[23].y

            left_ans =2*m.degrees(m.atan(dist(left_hip_x,left_hip_y,left_knee_x,left_knee_y)/dist(left_knee_x,left_knee_y,left_heel_x,left_heel_y)))
            right_ans = 2*m.degrees(m.atan(dist(right_hip_x,right_hip_y,right_knee_x,right_knee_y)/dist(right_knee_x,right_knee_y,right_heel_x,right_heel_y)))
            frame = cv2.copyMakeBorder(frame, 30, 30, 0, 0, cv2.BORDER_CONSTANT, black)

            print(left_ans)
            print(right_ans)
            
            # # PRINTING ANGLE AND ERROR DETECTION
            if left_ans >= 80 and left_ans <= 100:
                cv2.putText(frame,"Left Knee : " + str(round(left_ans,1)), 
                            tuple(np.multiply([left_knee_x,left_knee_y], [640, 480]).astype(int)), 
                            2,0.5, green, 1
                                    )
            else:
                cv2.putText(frame,"Left Knee" +str(round(left_ans,1)), 
                            tuple(np.multiply([left_knee_x,left_knee_y], [640, 480]).astype(int)), 
                            2,0.5,green,1
                                    )

            if right_ans >= 80 and right_ans <= 100:
                cv2.putText(frame, "Right Knee"+str(round(right_ans,1)), 
                            tuple(np.multiply([right_knee_x,right_knee_y], [640, 480]).astype(int)), 
                            2,0.5,green,1
                                    )
            else:
                cv2.putText(frame, "Right Knee"+str(round(right_ans,1)), 
                            tuple(np.multiply([right_knee_x,right_knee_y], [640, 480]).astype(int)), 
                            2,0.5,red,1
                                    )
        
            mp_drawing.draw_landmarks(frame,results.pose_landmarks,mp_holisitic.POSE_CONNECTIONS)
            # cv2.imshow("Goddess Posture" , frame)
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break

            ret, buffer = cv2.imencode('.jpg', frame)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')  # concat frame one by one and show result

    cap.release()
    cv2.destroyAllWindows()