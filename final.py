import cv2
from ultralytics import YOLO
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Category IDs for lost items
lost_items_categories = [
    1, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39,
    40, 41, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58, 59, 60,
    63, 64, 65, 66, 67, 73, 74, 75, 76, 77, 78, 79
]

# Directories for storing outputs
lost_item_folder = "LostItem"
fall_detected_folder = "FallDetected"
maintenance_required_folder = "MaintenanceRequired"

# Ensure directories exist
for folder in [lost_item_folder, fall_detected_folder, maintenance_required_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def bbox_center_distance(bbox1, bbox2):
    center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
    center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
    return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

MIN_DISTANCE = 60  # Minimum distance to consider for lost item proximity to person
DETECT_FACTOR = 3  # Threshold for consecutive detections before action

# Load YOLO models
model = YOLO('yolov8s.pt')
model2 = YOLO('BrokenTile.pt')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Video properties for output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

# Counters for consecutive detections
lost_item_counter = 0
fall_counter = 0
broken_tile_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection (converting from BGR to RGB)
    results = model([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)])
    results2 = model2([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)])

    # Detections for the first model
    detections = results[0].pred
    detections2 = results2[0].pred  # Detections for the second model

    # Variables to track detections
    potential_lost_item_detected = False
    fall_detected = False
    broken_tile_detected = False

    # Process first model detections
    for *xyxy, conf, cls in detections:
        if cls in lost_items_categories and conf > 0.5:
            if all(bbox_center_distance(xyxy, person_xyxy) >= MIN_DISTANCE for person_xyxy in detections if person_xyxy[5] == 0):
                potential_lost_item_detected = True
                break

    # Process fall detection based on person aspect ratio
    for *xyxy, conf, cls in detections:
        if cls == 0 and conf > 0.5:  # Assuming class 0 is 'person'
            aspect_ratio = (xyxy[3] - xyxy[1]) / (xyxy[2] - xyxy[0])
            if aspect_ratio < 0.6:
                fall_detected = True
                break

    # Process second model detections for maintenance
    for *xyxy, conf, cls in detections2:
        if cls == 0 and conf > 0.8:  # Assuming class 0 is 'broken tile'
            broken_tile_detected = True
            break

    # Update detection counters
    if potential_lost_item_detected:
        lost_item_counter += 1
    else:
          lost_item_counter = 0  # Reset counter if no lost item detected this frame

    if fall_detected:
        fall_counter += 1
    else:
        fall_counter = 0  # Reset counter if no fall detected this frame

    if broken_tile_detected:
        broken_tile_counter += 1
    else:
        broken_tile_counter = 0  # Reset counter if no broken tile detected this frame

    # Check if any of the counters reach the DETECT_FACTOR threshold to save images
    if lost_item_counter >= DETECT_FACTOR:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cv2.imwrite(os.path.join(lost_item_folder, f"LostItem_{timestamp}.png"), frame)
        lost_item_counter = 0  # Optionally reset counter after saving

    if fall_counter >= DETECT_FACTOR:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cv2.imwrite(os.path.join(fall_detected_folder, f"FallDetected_{timestamp}.png"), frame)
        fall_counter = 0  # Optionally reset counter after saving

    if broken_tile_counter >= DETECT_FACTOR:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cv2.imwrite(os.path.join(maintenance_required_folder, f"MaintenanceRequired_{timestamp}.png"), frame)
        broken_tile_counter = 0  # Optionally reset counter after saving

    # Write the frame into the output file
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('YOLO Real-time Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and writer, close all windows
cap.release()
out.release()
cv2.destroyAllWindows()
