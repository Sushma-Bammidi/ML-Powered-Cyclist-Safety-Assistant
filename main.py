import cv2
from rear_detection import RearVehicleDetector

detector = RearVehicleDetector()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("System Started. Press ESC to exit.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    processed_frame, alert = detector.process_frame(frame)

    if alert:
        print("Vehicle Alert Triggered")

    cv2.imshow("Cyclist Safety System", processed_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()