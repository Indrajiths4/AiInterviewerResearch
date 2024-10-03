import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def calculate_eye_aspect_ratio(eye_landmarks, face_landmarks):
    def distance(p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    vertical1 = distance(face_landmarks[eye_landmarks[1]], face_landmarks[eye_landmarks[5]])
    vertical2 = distance(face_landmarks[eye_landmarks[2]], face_landmarks[eye_landmarks[4]])
    horizontal = distance(face_landmarks[eye_landmarks[0]], face_landmarks[eye_landmarks[3]])
    
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def detect_emotion(face_landmarks):
    # Landmark indices
    left_eye = [33, 160, 158, 133, 153, 144]
    right_eye = [362, 385, 387, 263, 373, 380]
    left_eyebrow = [65, 55, 52, 53, 46]
    right_eyebrow = [295, 285, 282, 283, 276]
    mouth = [61, 291, 39, 181, 0, 17]

    # Calculate metrics
    left_ear = calculate_eye_aspect_ratio(left_eye, face_landmarks)
    right_ear = calculate_eye_aspect_ratio(right_eye, face_landmarks)
    avg_ear = (left_ear + right_ear) / 2.0

    eyebrow_height = (face_landmarks[left_eyebrow[2]].y + face_landmarks[right_eyebrow[2]].y) / 2
    mouth_height = face_landmarks[mouth[3]].y - face_landmarks[mouth[0]].y

    # Emotion detection logic
    if eyebrow_height < 0.3:
        if mouth_height > 0.1:
            return "Angry"
        else:
            return "Stressed"
    elif avg_ear < 0.2:
        return "Tension"
    elif eyebrow_height > 0.4:
        if mouth_height < 0.05:
            return "Fear"
        else:
            return "Surprised"
    elif mouth_height > 0.15:
        return "Happy"
    else:
        return "Neutral"

def main():
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh, mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            face_results = face_mesh.process(image)
            pose_results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    
                    emotion = detect_emotion(face_landmarks.landmark)
                    cv2.putText(image, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                
                if abs(left_shoulder.y - right_shoulder.y) < 0.05:
                    posture = "Good posture"
                else:
                    posture = "Improve posture"
                
                cv2.putText(image, f"Posture: {posture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('MediaPipe Interview Demo', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()