import cv2
from ultralytics import YOLO
import datetime

# 定义可能会在机场遗失的物品类别
lost_items_categories = [
    1, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39,
    40, 41, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60,
    63, 64, 65, 66, 67, 73, 74, 75, 76, 77, 78, 79
]

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize a simple counter for the number of people detected in each frame
Person_count_per_frame = []

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection (converting from BGR to RGB)
    results = model([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)])
    result = results[0].cpu()  # Ensure the results are on CPU

    # Initialize the count of people for the current frame and lost item flag
    current_frame_count = 0
    lost_item_detected = False
    fall_detected = False  # Assuming you have a mechanism or a condition to detect a fall

    # Check if detections exist
    if len(result.boxes) > 0:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls_idx = box
            label = result.names[int(cls_idx)]
            score = conf

            # Convert coordinates to integers
            top_left = int(x1), int(y1)
            bottom_right = int(x2), int(y2)

            # Draw the bounding box and label
            text_color = (36, 255, 12)
            if cls_idx == 0 and score >= 0.5:  # 'person' class
                current_frame_count += 1
                aspect_ratio = (y2 - y1) / (x2 - x1)
                # 判断是否跌倒
                if aspect_ratio < 0.5:
                    color = (0, 0, 255)  # Red
                    text_color = (0, 0, 255)
                    fall_detected = True
                else:
                    color = (0, 255, 0)  # Green

            elif cls_idx in lost_items_categories and current_frame_count == 0:
                lost_item_detected = True
                color = (0, 255, 255)  # Yellow for lost items if no person detected
                text_color = (0, 255, 255)
            else:
                color = (0, 255, 0)  # Green for other detections

            cv2.rectangle(frame, top_left, bottom_right, color, 2)
            label_with_score = f"{label}: {score:.2f}"
            cv2.putText(frame, label_with_score, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color,
                        2)

    # Display the count of people in the current frame on the video
    cv2.putText(frame, f'Person Count: {current_frame_count}', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # If lost items are detected but no person is detected, display a warning
    if lost_item_detected:
        cv2.putText(frame, "Lost Item Detected!", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        print('Lost Item Detected!')
        print('Event Level: 3')

    # Check for fall detection
    if fall_detected:
        cv2.putText(frame, 'Person Fall Detected!', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        print('Person Fall Detected!')
        print('Event Level: 1')

    # Record the count of people per frame
    Person_count_per_frame.append(current_frame_count)

    # Display the resulting frame
    cv2.imshow('YOLO Real-time Detection', frame)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()

# Print or log the number of people data
print("Person count per frame:", Person_count_per_frame)
# Data can also be saved to a file
with open("Person_count_log.txt", "w") as file:
    for count in Person_count_per_frame:
        file.write(f"{datetime.datetime.now()}, {count}\n")
