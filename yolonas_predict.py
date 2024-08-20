"""
Run Yolo-NAS model on a video to get person/vehicle detections and visualize the results
"""

import cv2
from ultralytics import NAS

# Load the Yolo-NAS model
model = NAS("yolo_nas_m.pt")

# Open the video file
video_path = "video.mkv"

cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run Yolo-NAS inference on the frame
        # Class 0 = person, 2 = car, 5 = bus, 7 = truck
        results = model(frame, classes=[0,2,5,7], verbose=False, conf=0.35)

        # Run Yolo-NAS inference and Bytetrack tracker on the frame
        #results = model.track(frame, persist=True, classes=[0,2,5,7], verbose=False, conf=0.35)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        annotated_frame = cv2.resize(annotated_frame, (1280, 720))
        cv2.imshow("YOLONAS Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()