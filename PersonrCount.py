import cv2
from ultralytics import YOLO
import datetime

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

    # Initialize the count of people for the current frame
    current_frame_count = 0

    # Check if detections exist
    if len(result.boxes) > 0:
        for box in result.boxes.data:
            # Parse the coordinates and category information of the bounding box
            x1, y1, x2, y2, conf, cls_idx = box
            label = result.names[int(cls_idx)]
            score = conf

            # Count the label 'person' with a confidence threshold of 0.5
            if label == 'person' and score >= 0.5:
                current_frame_count += 1

                # Convert coordinates to integers
                top_left = int(x1), int(y1)
                bottom_right = int(x2), int(y2)

                # Draw the bounding box and label
                cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
                label_with_score = f"{label}: {score:.2f}"
                cv2.putText(frame, label_with_score, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (36, 255, 12), 2)

    # Display the count of people in the current frame on the video
    cv2.putText(frame, f'Person Count: {current_frame_count}', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)

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
