import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection (converting BGR to RGB)
    results = model([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)])
    result = results[0].cpu()  # 确保结果在 CPU 上

    # Check if detections exist
    if len(result.boxes) > 0:
        for box in result.boxes.data:
            # 解析边界框坐标和类别信息
            x1, y1, x2, y2, conf, cls_idx = box
            label = result.names[int(cls_idx)]
            score = conf

            # 转换坐标为整数
            top_left = int(x1), int(y1)
            bottom_right = int(x2), int(y2)

            # 绘制边界框和标签
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
            label_with_score = f"{label}: {score:.2f}"
            cv2.putText(frame, label_with_score, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    else:
        print("No detections")

    # Display the resulting frame
    cv2.imshow('YOLO Real-time Detection', frame)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
