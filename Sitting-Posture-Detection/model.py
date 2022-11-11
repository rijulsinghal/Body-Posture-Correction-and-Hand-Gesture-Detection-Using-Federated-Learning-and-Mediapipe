import mediapipe as mp
import math as m
import cv2
time_string_good = None
time_string_bad = None
                    
def findAngle(x1, y1, x2, y2):
    theta = m.acos( (y2 -y1)*(-y1) / (m.sqrt(
        (x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
    degree = int(180/m.pi)*theta
    return degree


def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist


    
def main_func():    
    mp_holisitic = mp.solutions.holistic
    cap = cv2.VideoCapture(0)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    good_frames = 0
    bad_frames  = 0
    global time_string_good,time_string_bad

    with mp_holisitic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            sucess, frame = cap.read()
            h, w = frame.shape[:2]
            
            if not sucess:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                image = buffer.tobytes()
                results = holistic.process(frame)
                if(results.pose_landmarks is None):
                    continue

                left_shoulder_y = results.pose_landmarks.landmark[12].y*h
                left_shoulder_z = results.pose_landmarks.landmark[12].z*w

                left_ear_y = results.pose_landmarks.landmark[8].y*h
                left_ear_z = results.pose_landmarks.landmark[8].z*w

                min_dis = m.degrees(m.atan(abs(left_shoulder_z-left_ear_z)/abs(left_ear_y-left_shoulder_y)))

                if (min_dis <= 60):
                    good_frames += 1
                    good_time = (1 / fps) * good_frames
                    time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
                    fp = open("Sitting-Posture-Detection/good_time.txt","w")
                    fp.write(time_string_good)
                    # print(time_string_good)
                    fp.close()
                
                else:
                    bad_frames += 1
                    bad_time =  (1 / fps) * bad_frames
                    time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
                    fp = open("Sitting-Posture-Detection/bad_time.txt","w")
                    fp.write(time_string_bad)
                    # print(time_string_bad)
                    fp.close()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')  # concat frame one by one and show result
                
