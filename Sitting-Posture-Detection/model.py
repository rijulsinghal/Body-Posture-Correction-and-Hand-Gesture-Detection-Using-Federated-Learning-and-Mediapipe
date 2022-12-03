import mediapipe as mp
import math as m
import cv2
time_string_good = 'Good Posture Time: 00:00:00.'
time_string_bad = 'Bad Posture Time : 00:00:00.'

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
white = (255,255,255)

def findAngle(x1, y1, x2, y2):
    theta = m.acos( (y2 -y1)*(-y1) / (m.sqrt(
        (x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
    degree = int(180/m.pi)*theta
    return degree

def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
     
    return "%d:%02d:%02d" % (hour, minutes, seconds)

def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist
    
def main_func(mode=None):    
    mp_holisitic = mp.solutions.holistic
    cap = cv2.VideoCapture(0)
    good_frames = 0
    bad_frames  = 0
    good_time = 0
    angle = 50
    bad_time = 0
    global time_string_good,time_string_bad    
    time_string_good = 'Good Posture Time: 00:00:00.'
    time_string_bad = 'Bad Posture Time : 00:00:00.'

    fp = open("Sitting-Posture-Detection/accuracy.txt" , "w")
    fp.write(str(0))
    fp.close()

    with mp_holisitic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            sucess, frame = cap.read()
            fps = int(cap.get(cv2.CAP_PROP_FPS))
   
            if not sucess:
                break
            else:
                h, w = frame.shape[:2]
                results = holistic.process(frame)
                if(results.pose_landmarks is None):
                    continue

                left_shoulder_y = results.pose_landmarks.landmark[12].y*h
                left_shoulder_z = results.pose_landmarks.landmark[12].z*w

                left_ear_y = results.pose_landmarks.landmark[8].y*h
                left_ear_z = results.pose_landmarks.landmark[8].z*w

                if(mode != None):
                    if(mode == "Lenient"):
                        angle = 50
                    elif(mode == "Medium"):
                        angle = 48
                    elif(mode == "Strict"):
                        angle = 47

                    min_dis = m.degrees(m.atan(abs(left_shoulder_z-left_ear_z)/abs(left_ear_y-left_shoulder_y)))
                    
                    # print(angle)
                    # print(min_dis)  

                    if (min_dis <= angle):
                        good_frames += 1
                        
                    else:
                        bad_frames += 1
                        
                    good_time = (1 / fps) * good_frames
                    bad_time =  (1 / fps) * bad_frames
                    time_string_good = 'Good Posture Time : ' + str(convert(good_time*3)) + '.'
                    time_string_bad = 'Bad Posture Time : ' + str(convert(bad_time*3)) + '.'
                
                fp = open("Sitting-Posture-Detection/accuracy.txt" , "w")
                if(bad_time == 0):
                    fp.write(str(100))
                else:
                    fp.write(str(good_time/(good_time+bad_time)*100))

                fp.close()
                frame = cv2.copyMakeBorder(frame, 30, 30, 30, 30, cv2.BORDER_CONSTANT, None, value = white)

                cv2.putText(frame, time_string_good, (30,25), font, 0.6, dark_blue, 2)
                cv2.putText(frame, time_string_bad, (400,25), font, 0.6, red, 2)
                
                cv2.imshow('Sitting-Posture-Correction', frame)
                # cv2.imshow("Bitilasana Pose", frame)
                if (cv2.waitKey(10) & 0xFF == ord('q')):
                    break

                ret, buffer = cv2.imencode('.jpg', frame)
                image = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')  # concat frame one by one and show result
    cap.release()
    cv2.destroyAllWindows()
