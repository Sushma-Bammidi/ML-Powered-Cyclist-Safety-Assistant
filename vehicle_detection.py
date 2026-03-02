from ultralytics import YOLO
import cv2

# -----------------------------
# Load Model
# -----------------------------
print("Loading YOLOv8 model...")
model = YOLO("yolov8s.pt")  # more accurate than yolov8n
print("Model loaded successfully.")

# -----------------------------
# Initialize Camera
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

# Improve resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# -----------------------------
# Memory for Tracking Areas
# -----------------------------
previous_areas = {}

print("Starting detection... Press ESC to exit.")

# -----------------------------
# Main Loop
# -----------------------------
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Use tracking for stable IDs
    results = model.track(
        frame,
        persist=True,
        conf=0.5,
        verbose=False
    )

    for r in results:
        boxes = r.boxes

        if boxes.id is not None:
            for box, track_id in zip(boxes, boxes.id):

                cls = int(box.cls[0])
                label = model.names[cls]

                # Detect only vehicles
                if label in ["car", "bus", "truck", "motorcycle"]:

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    track_id = int(track_id)

                    approaching = False

                    # Check area growth
                    if track_id in previous_areas:
                        previous_area = previous_areas[track_id]
                        growth = area / (previous_area + 1)

                        if growth > 1.15:
                            approaching = True

                    previous_areas[track_id] = area

                    # Draw bounding box
                    color = (0, 0, 255) if approaching else (0, 255, 0)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    text = f"{label} ID:{track_id}"
                    cv2.putText(
                        frame,
                        text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

                    if approaching:
                        cv2.putText(
                            frame,
                            "WARNING: VEHICLE APPROACHING!",
                            (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3
                        )

    cv2.imshow("Cyclist Safety Assistant", frame)

    if cv2.waitKey(1) == 27:
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
print("Program stopped.")