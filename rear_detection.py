from ultralytics import YOLO
import cv2
import time

class RearVehicleDetector:

    def __init__(self):
        print("Loading YOLOv8s model...")
        self.model = YOLO("yolov8s.pt")
        print("Model loaded.")

        self.previous_data = {}
        self.alert_cooldown = 0

        self.vehicle_classes = ["car", "bus", "truck", "motorcycle"]

    def process_frame(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            conf=0.5,
            verbose=False
        )

        alert_triggered = False

        for r in results:
            boxes = r.boxes

            if boxes.id is not None:
                for box, track_id in zip(boxes, boxes.id):

                    cls = int(box.cls[0])
                    label = self.model.names[cls]

                    if label in self.vehicle_classes:

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)
                        center_y = (y1 + y2) // 2
                        track_id = int(track_id)

                        approaching = False

                        if track_id in self.previous_data:
                            prev_area, prev_center_y = self.previous_data[track_id]

                            area_growth = area / (prev_area + 1)
                            vertical_movement = center_y - prev_center_y

                            if area_growth > 1.05 and vertical_movement > 2:
                                approaching = True

                        self.previous_data[track_id] = (area, center_y)

                        color = (0, 0, 255) if approaching else (0, 255, 0)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        cv2.putText(frame,
                                    f"{label} ID:{track_id}",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    color,
                                    2)

                        if approaching and time.time() > self.alert_cooldown:
                            cv2.putText(frame,
                                        "WARNING: VEHICLE APPROACHING!",
                                        (40, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1,
                                        (0, 0, 255),
                                        3)

                            alert_triggered = True
                            self.alert_cooldown = time.time() + 2

        return frame, alert_triggered