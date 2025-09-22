import torch
import cv2

# Model​
model = torch.hub.load("ultralytics/yolov5", "yolov5m")

# Video capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame (BGR to RGB)
    ret, frame = cap.read()

    if not ret:
        print("Cannot read frame.")
        break

    # 추론 실행 (BGR -> RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)

    for i, obj in enumerate(results.xyxy[0]):
        x1, y1, x2, y2, _, cls = map(int, obj)
        conf = obj[4]
        label = f"{model.names[cls]} {conf:.2f}"
        print(f"Object {i}: {label} at [{x1}, {y1}, {x2}, {y2}]")

        # 사각형 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 라벨 텍스트
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv5 cam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

