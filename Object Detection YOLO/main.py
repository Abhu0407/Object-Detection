from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO('yolov8n.pt')


# 📸 Detect objects in image
def basic_object(url):
    results = model(url, show=True)

    # Print detected objects
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"Detected: {model.names[cls]} with confidence {conf:.2f}")


# 🎥 Real-time detection from webcam
def webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("YOLOv8 Real-Time Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# 🎞️ Detect objects in a video file
def detect_object_in_video(url):
    results = model(url, show=True, save=True)
    print(f"Processed video saved in 'runs/detect/predict' folder.")


# 🚀 Main menu
if __name__ == "__main__":
    while True:
        print("\n===== YOLO Object Detection =====")
        print("1. Detect objects in an image")
        print("2. Real-time detection (Webcam)")
        print("3. Detect objects in a video")
        print("4. Exit")

        choice = input("Enter your choice (1/2/3/4): ")

        if choice == "1":
            path = input("Enter image path: ")
            basic_object(path)

        elif choice == "2":
            webcam()

        elif choice == "3":
            path = input("Enter video path: ")
            detect_object_in_video(path)

        elif choice == "4":
            print("Exiting...")
            break

        else:
            print("Invalid choice! Please select again.")
