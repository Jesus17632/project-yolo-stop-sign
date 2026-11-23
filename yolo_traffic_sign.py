import cv2
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame)
        cv2.imshow("Detección de Señales", results[0].plot())
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()