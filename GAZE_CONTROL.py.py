import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

while True:
    _, frame = cap.read()
    new_frame=np.zeros((500,500,3),np.uint8) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)

        # Detect blinking
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 3:
            cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))


            # Gaze detection
            left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                        (landmarks.part(37).x, landmarks.part(37).y),
                                        (landmarks.part(38).x, landmarks.part(38).y),
                                        (landmarks.part(39).x, landmarks.part(39).y),
                                        (landmarks.part(40).x, landmarks.part(40).y),
                                        (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
            #cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

            height, width, _ = frame.shape
            mask = np.zeros((height, width), np.uint8)
            cv2.polylines(mask, [left_eye_region], True, 255, 2)
            cv2.fillPoly(mask, [left_eye_region], 255)
            eye = cv2.bitwise_and(gray, gray, mask=mask)


            min_x = np.min(left_eye_region[:, 0])
            max_x = np.max(left_eye_region[:, 0])
            min_y = np.min(left_eye_region[:, 1])
            max_y = np.max(left_eye_region[:, 1])

            gray_eye = eye[min_y: max_y, min_x: max_x]
            _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
            height,width=threshold_eye.shape
            left_side_threshold=threshold_eye[0:height,0:int(width/2)]
            left_side_white=cv2.countNonZero(left_side_threshold)
            right_side_threshold=threshold_eye[0:height,0:int(width/2)]
            right_side_white=cv2.countNonZero(right_side_threshold)
            if left_side_white==0:
                gaze_ratio=1
            elif right_side_white==0:
                    gaze_ratio=5
            else:
                gaze_ratio=(left_side_white/right_side_white)/2
                       
                if gaze_ratio <= 0.01:
                            cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
                            new_frame[:] = (0, 0, 255)
                elif 0.01 < gaze_ratio < 0.3:
                                cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
                else:
                    new_frame[:] = (255, 0, 0)
                    cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)
                                    

                    cv2.putText(frame,str(gaze_ratio),(30,100),font,2,(0,0,255),3)

                    threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
                    eye = cv2.resize(gray_eye, None, fx=5, fy=5)
                    cv2.imshow("Threshold", threshold_eye)
                    cv2.imshow("left",left_side_threshold)
                    cv2.imshow("right",right_side_threshold)


                    cv2.imshow("Frame", frame)
                    cv2.imshow("New frame",new_frame)
                    key = cv2.waitKey(1)
                    if key == ord("e"):
                        break

                    cap.release()
                    cv2.destroyAllWindows()
