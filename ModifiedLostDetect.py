import cv2
from ultralytics import YOLO
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

lost_items_categories = [
    1, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39,
    40, 41, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58, 59, 60,
    63, 64, 65, 66, 67, 73, 74, 75, 76, 77, 78, 79
]
# 56: chair

DETECT_FACTOR = 3

lost_item_folder = "LostItem"
fall_detected_folder = "FallDetected"
maintenance_required_folder = "MaintenanceRequired"

# 确保文件夹存在，如果不存在，则创建
for folder in [lost_item_folder, fall_detected_folder, maintenance_required_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def bbox_center_distance(bbox1, bbox2):
    center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
    center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance

MIN_DISTANCE = 60

# Load the YOLO model
model = YOLO('yolov8s.pt')
model2 = YOLO('BrokenTile.pt')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize a simple counter for the number of people detected in each frame
Person_count_per_frame = []

# Initialize a list for storing the cumulative number of people detected
cumulative_person_count = []
timestamps = []
plt.style.use('fivethirtyeight')
# Initialize a frame counter
frame_counter = 0

# Get video properties to use with the video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection (converting from BGR to RGB)
    results = model([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)])
    result = results[0].cpu()  # Ensure the results are on CPU

    # Perform detection with the second model
    results2 = model2([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)])
    result2 = results2[0].cpu()  # Ensure the results are on CPU

    # Initialize the count of people for the current frame and lost item flag
    current_frame_count = 0
    potential_lost_item_detected = False
    fall_detected = False  # Assuming you have a mechanism or a condition to detect a fall
    broken_tile_detected = False

    # Process detections from the second model
    if len(result2.boxes) > 0:
        for box in result2.boxes.data:
            x1, y1, x2, y2, conf, cls_idx = box
            # Assuming the class index for "Broken Tile" in your model is 0
            # Adjust if your model has a different index for this class
            if cls_idx == 0 and conf >= 0.8:
                broken_tile_detected = True
                # Highlight broken tiles with a specific color, e.g., purple
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
                cv2.putText(frame, f"Broken Tile: {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 0, 255), 2)

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
                if aspect_ratio < 0.6:  # 0.6 means: height/width < 0.6
                    color = (0, 0, 255)  # Red
                    text_color = (0, 0, 255)
                    fall_detected = True
                else:
                    color = (0, 255, 0)  # Green
            elif cls_idx in lost_items_categories:
                # 检查物品周围是否有人
                is_lost = True
                for person_box in result.boxes.data:
                    _, _, _, _, _, cls_idx = person_box  # 解包边界框信息，获取类别索引
                    if cls_idx == 0:  # 假设类别索引0代表人物
                        if bbox_center_distance(box, person_box) < MIN_DISTANCE:
                            is_lost = False
                            break
                if is_lost:
                    potential_lost_item_detected = True
                    color = (0, 255, 255)  # Yellow for lost items
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
    # Record the count of people per frame
    Person_count_per_frame.append(current_frame_count)

    # Write the frame into the file 'output.avi'
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('YOLO Real-time Detection', frame)

    # Append the current frame's person count to the cumulative list
    cumulative_person_count.append(current_frame_count)
    timestamps.append(datetime.datetime.now())
    frame_counter += 1

    # Check if the frame counter is a multiple of 30
    if frame_counter % 30 == 0:
        plt.figure(figsize=(12, 7))
        # Make sure to convert timestamps to matplotlib date format
        dates = mdates.date2num(timestamps)
        plt.plot_date(dates, cumulative_person_count, marker='', linestyle='-', color='royalblue', linewidth=2.5)
        plt.title('Person Volume Log', fontsize=18, fontweight='bold')
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Number of People', fontsize=14)
        # Format the date axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gca().xaxis.set_major_locator(mdates.MinuteLocator())
        plt.gca().xaxis.set_minor_locator(mdates.SecondLocator(interval=30))
        plt.gcf().autofmt_xdate()  # Beautify the x-labels
        plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray')
        plt.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray')
        plt.tight_layout()  # Adjust the layout to make room for the rotated x-axis labels
        plt.savefig('cumulative_flow_chart.png')  # This will overwrite the existing file
        plt.close()

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()

# Print or log the number of people data
print("Person count per frame:", Person_count_per_frame)
# Data can also be saved to a file
with open("Person_count_log.txt", "w") as file:
    for count in Person_count_per_frame:
        file.write(f"{datetime.datetime.now()}, {count}\n")