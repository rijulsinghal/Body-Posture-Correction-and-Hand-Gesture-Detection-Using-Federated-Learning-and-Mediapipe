# Import Requried Libraries
import numpy as np
import mediapipe as mp
import math as m
import cv2

# Colours to be Used
red = (50, 50, 255)
green = (127, 255, 0)
black = (0, 0, 0)

# Window Size of the Frame
window_size = [640, 480]

# Calculate Angle of a Certain Joint using 3 Landmarks
def calculate_angle(x1, y1, x2, y2, x3, y3):
    # # Find inverse tangent of the distances between the points from the required joint
    angle = m.degrees(m.atan2(y3 - y2, x3 - x2) - m.atan2(y1 - y2, x1 - x2))
    # # Convert negative angle to its positive equivalent
    if (angle < 0):
        angle += 360 
    return angle

# MediaPipe Initialization
def main():
    mp_holisitic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    # Start Video Capture
    cap = cv2.VideoCapture(1)
    with mp_holisitic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:

        while cap.isOpened():
            # Open Current Frame to Work on
            ret, frame = cap.read()
            results = holistic.process(frame)
        
            if(results.pose_landmarks is None):
                continue
            
            # Obtaining Landmark Coordinates for Parivrtta Trikonasana
            # # Shoulder Coordinates
            left_shoulder_x = results.pose_landmarks.landmark[11].x
            left_shoulder_y = results.pose_landmarks.landmark[11].y
            right_shoulder_x = results.pose_landmarks.landmark[12].x
            right_shoulder_y = results.pose_landmarks.landmark[12].y
            # # Elbow Coordinates
            left_elbow_x = results.pose_landmarks.landmark[13].x
            left_elbow_y = results.pose_landmarks.landmark[13].y
            right_elbow_x = results.pose_landmarks.landmark[14].x
            right_elbow_y = results.pose_landmarks.landmark[14].y
            # # Wrist Coordinates
            left_wrist_x = results.pose_landmarks.landmark[15].x
            left_wrist_y = results.pose_landmarks.landmark[15].y
            right_wrist_x = results.pose_landmarks.landmark[16].x
            right_wrist_y = results.pose_landmarks.landmark[16].y
            # # Hip Coordinates
            left_hip_x = results.pose_landmarks.landmark[23].x
            left_hip_y = results.pose_landmarks.landmark[23].y
            right_hip_x = results.pose_landmarks.landmark[24].x
            right_hip_y = results.pose_landmarks.landmark[24].y
            # # Knee Coordinates
            left_knee_x = results.pose_landmarks.landmark[25].x
            left_knee_y = results.pose_landmarks.landmark[25].y
            right_knee_x = results.pose_landmarks.landmark[26].x
            right_knee_y = results.pose_landmarks.landmark[26].y
            # # Ankle Coordinates
            left_ankle_x = results.pose_landmarks.landmark[27].x
            left_ankle_y = results.pose_landmarks.landmark[27].y
            right_ankle_x = results.pose_landmarks.landmark[28].x
            right_ankle_y = results.pose_landmarks.landmark[28].y
                    
            # Calculating the Angles required for Parivrtta Trikonasana
            left_elbow_angle = calculate_angle(left_shoulder_x, left_shoulder_y, left_elbow_x, left_elbow_y, left_wrist_x, left_wrist_y)
            right_elbow_angle = calculate_angle(right_shoulder_x, right_shoulder_y, right_elbow_x, right_elbow_y, right_wrist_x, right_wrist_y)
            left_shoulder_angle = calculate_angle(left_elbow_x, left_elbow_y, left_shoulder_x, left_shoulder_y, left_hip_x, left_hip_y)
            right_shoulder_angle = calculate_angle(right_elbow_x, right_elbow_y, right_shoulder_x, right_shoulder_y, right_hip_x, right_hip_y)
            left_hip_angle = calculate_angle(left_shoulder_x, left_shoulder_y, left_hip_x, left_hip_y, left_knee_x, left_knee_y)
            right_hip_angle = calculate_angle(right_shoulder_x, right_shoulder_y, right_hip_x, right_hip_y, right_knee_x, right_knee_y)
            left_knee_angle = calculate_angle(left_hip_x, left_hip_y, left_knee_x, left_knee_y, left_ankle_x, left_ankle_y)
            right_knee_angle = calculate_angle(right_hip_x, right_hip_y, right_knee_x, right_knee_y, right_ankle_x, right_ankle_y)
            
            # Adding border to the frame
            frame = cv2.copyMakeBorder(frame, 30, 30, 0, 0, cv2.BORDER_CONSTANT, black)
                
            # Check if Parivrtta Trikonasana Conditions are met
            # # Check if Elbows are Straight
            # # # Check if Left Elbow is Straight
            if (left_elbow_angle > 170) and (left_elbow_angle < 190):
                cv2.putText(frame, 'Elbow Angle: ' + str(round(left_elbow_angle, 1)),
                            tuple(np.multiply([left_elbow_x, left_elbow_y], window_size).astype(int)), 
                            2, 0.5, green, 1)
            else:
                cv2.putText(frame, 'Elbow Angle: ' + str(round(left_elbow_angle, 1)), 
                            tuple(np.multiply([left_elbow_x, left_elbow_y], window_size).astype(int)),
                            2, 0.5, red, 1)
            # # # Check if Right Elbow is Straight
            if (right_elbow_angle > 170) and (right_elbow_angle < 190):
                cv2.putText(frame, 'Elbow Angle: ' + str(round(right_elbow_angle, 1)),
                            tuple(np.multiply([right_elbow_x, right_elbow_y], window_size).astype(int)),
                            2, 0.5, green, 1)
            else:
                cv2.putText(frame, 'Elbow Angle: ' + str(round(right_elbow_angle, 1)), 
                            tuple(np.multiply([right_elbow_x, right_elbow_y], window_size).astype(int)),
                            2, 0.5, red, 1)
            # # Check if Shoulders are in the Correct Position
            # # # Check Left Shoulder
            if (left_shoulder_angle > 75) and (left_shoulder_angle < 105):
                cv2.putText(frame, 'Shoulder Angle: ' + str(round(left_shoulder_angle, 1)),
                            tuple(np.multiply([left_shoulder_x, left_shoulder_y], window_size).astype(int)),
                            2, 0.5, green, 1)
            else:
                cv2.putText(frame, 'Shoulder Angle: ' + str(round(left_shoulder_angle, 1)), 
                            tuple(np.multiply([left_shoulder_x, left_shoulder_y], window_size).astype(int)),
                            2, 0.5, red, 1)
            # # # Check Right Shoulder
            if (right_shoulder_angle > 75) and (right_shoulder_angle < 105):
                cv2.putText(frame, 'Shoulder Angle: ' + str(round(right_shoulder_angle, 1)),
                            tuple(np.multiply([right_shoulder_x, right_shoulder_y], window_size).astype(int)),
                            2, 0.5, green, 1)
            else:
                cv2.putText(frame, 'Shoulder Angle: ' + str(round(right_shoulder_angle, 1)), 
                            tuple(np.multiply([right_shoulder_x, right_shoulder_y], window_size).astype(int)),
                            2, 0.5, red, 1)
            # # Check if Hips are in the Correct Position
            # # # Check Left Hip Coordinate
            if ((left_hip_angle > 50) and (left_hip_angle < 80)) and ((right_hip_angle > 100) and (right_hip_angle < 130)) or ((left_hip_angle > 100) and (left_hip_angle < 130)) or ((right_hip_angle > 50) and (right_hip_angle < 80)):
                cv2.putText(frame, 'Hip Angle: ' + str(round(left_hip_angle, 1)), 
                            tuple(np.multiply([left_hip_x, left_hip_y], window_size).astype(int)),
                            2, 0.5, green, 1)
                cv2.putText(frame, 'Hip Angle: ' + str(round(right_hip_angle, 1)), 
                            tuple(np.multiply([right_hip_x, right_hip_y], window_size).astype(int)),
                            2, 0.5, green, 1)
            # # # If Hips not in Position
            else:
                cv2.putText(frame, 'Hip Angle: ' + str(round(left_hip_angle, 1)), 
                            tuple(np.multiply([left_hip_x, left_hip_y], window_size).astype(int)),
                            2, 0.5, red, 1)
                cv2.putText(frame, 'Hip Angle: ' + str(round(right_hip_angle, 1)), 
                            tuple(np.multiply([right_hip_x, right_hip_y], window_size).astype(int)),
                            2, 0.5, red, 1)
            # # Check if Knees are in the Correct Position
            # # # Check Left Knee Coordinate
            if (left_knee_angle > 170) and (left_knee_angle < 190):
                cv2.putText(frame, 'Knee Angle: ' + str(round(left_knee_angle, 1)),
                            tuple(np.multiply([left_knee_x, left_knee_y], window_size).astype(int)),
                            2, 0.5, green, 1)
            else:
                cv2.putText(frame, 'Knee Angle: ' + str(round(left_knee_angle, 1)), 
                            tuple(np.multiply([left_knee_x, left_knee_y], window_size).astype(int)),
                            2, 0.5, red, 1)
            # # # Check Right Knee Coordinate
            if (right_knee_angle > 170) and (right_knee_angle < 190):
                cv2.putText(frame, 'Knee Angle: ' + str(round(right_knee_angle, 1)),
                            tuple(np.multiply([right_knee_x, right_knee_y], window_size).astype(int)),
                            2, 0.5, green, 1)
            else:
                cv2.putText(frame, 'Knee Angle: ' + str(round(right_knee_angle, 1)), 
                            tuple(np.multiply([right_knee_x, right_knee_y], window_size).astype(int)),
                            2, 0.5, red, 1)
            
            # Output
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holisitic.POSE_CONNECTIONS)
            # cv2.imshow("Parivrtta Trikonasana Pose", frame)
            # if (cv2.waitKey(10) & 0xFF == ord('q')):
            #     break
            ret, buffer = cv2.imencode('.jpg', frame)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')  # concat frame one by one and show result

    # End Video Capture
    cap.release()
    cv2.destroyAllWindows()
