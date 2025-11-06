from ultralytics import YOLO
import cv2
import os
import keyboard

# Load YOLOv8 model
model = YOLO('yolov8n.pt')


def basic_object(image_path):
    # Check if image exists
    if not os.path.exists(image_path):
        print("❌ Error: Image not found. Please check your path.")
        return

    # Run YOLO detection
    results = model(image_path)

    # Get result image (with boxes drawn)
    for r in results:
        annotated_frame = r.plot()  # Draws boxes and labels

        # Save output image
        output_path = "output_detected_image.jpg"
        cv2.imwrite(output_path, annotated_frame)

        # Show output
        cv2.imshow("Detected Objects", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("\n✅ Detection complete!")
        print(f"📸 Output saved at: {os.path.abspath(output_path)}")

        # Print all detected objects
        print("\n🧠 Detected Objects:")
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"• {model.names[cls]} ({conf:.2f})")


def webcam():
    print("\n🚀 Starting real-time webcam detection...")
    print("🎮 Command: Press 'Q' in the terminal anytime to quit webcam detection.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ Error: Could not access the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame from webcam.")
            break

        # Run YOLO detection silently
        results = model(frame, stream=True, verbose=False)

        # Draw detections
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display frame
        cv2.imshow("YOLOv8 Webcam Detection", frame)

        # Quit if 'Q' is pressed (from anywhere)
        if keyboard.is_pressed('q') or keyboard.is_pressed('Q'):
            print("🛑 Webcam detection stopped by user.")
            break

        # Small delay to reduce CPU usage
        if cv2.waitKey(1) == 27:  # ESC fallback
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_object_in_video(video_path):
    if not os.path.exists(video_path):
        print("❌ Error: Video not found. Please check your path.")
        return

    results = model(video_path, show=True, save=True)
    print(f"\n🎞️ Video processed successfully!")
    print("📁 Output saved inside 'runs/detect/predict' folder.")


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
            print("👋 Exiting...")
            break

        else:
            print("❌ Invalid choice! Please try again.")
