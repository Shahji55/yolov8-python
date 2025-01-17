"""
Run Yolov8 model on a video to get vehicle detections and visualize the results
"""

import cv2
from ultralytics import YOLO

def predict():
    # Load the YOLOv8 model (default model is trained on COCO dataset)
    model = YOLO("yolov8s.pt", task='detect')

    # Specify the video source
    source = "vehicle.mp4"

    # Open the video file
    cap = cv2.VideoCapture(source)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            # Class 2 = car, 5 = bus, 7 = truck
            results = model(frame, classes=[2,5,7], verbose=False, conf=0.35)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            annotated_frame = cv2.resize(annotated_frame, (1280, 720))
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict()