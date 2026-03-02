import cv2
import mediapipe as mp
import numpy as np


class FatigueDetector:
    def __init__(self):
        # Use new MediaPipe FaceMesh API
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )

        self.drowsy_frames = 0

    def detect(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                # Eye landmark indexes
                LEFT_EYE = [33, 160, 158, 133, 153, 144]
                RIGHT_EYE = [362, 385, 387, 263, 373, 380]

                left_eye = []
                right_eye = []

                h, w, _ = frame.shape

                for idx in LEFT_EYE:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    left_eye.append((x, y))

                for idx in RIGHT_EYE:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    right_eye.append((x, y))

                # Draw eye points
                for point in left_eye + right_eye:
                    cv2.circle(frame, point, 2, (0, 255, 0), -1)

                # Simple vertical distance check
                left_vertical = np.linalg.norm(np.array(left_eye[1]) - np.array(left_eye[5]))
                right_vertical = np.linalg.norm(np.array(right_eye[1]) - np.array(right_eye[5]))

                if left_vertical < 5 and right_vertical < 5:
                    self.drowsy_frames += 1
                else:
                    self.drowsy_frames = 0

                if self.drowsy_frames > 20:
                    cv2.putText(frame,
                                "DROWSINESS ALERT!",
                                (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                3)

        return frame