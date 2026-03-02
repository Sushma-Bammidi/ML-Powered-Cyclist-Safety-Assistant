import cv2
from fatigue_detection import FatigueDetector

detector = FatigueDetector()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = detector.detect(frame)

    cv2.imshow("Fatigue Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()