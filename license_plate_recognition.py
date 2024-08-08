"""
Run Yolov8 model on a video to get license plate detections, 
use EasyOCR to recognize the number and visualize the results
"""

from ultralytics import YOLO
import easyocr
import cv2

def predict():
    # Load the YOLOv8 model
    model = YOLO("license_plate_detector.pt")

    # Specify the video source
    source = 'license_plate.mp4'

    # Open the video file
    cap = cv2.VideoCapture(source)

    # Initialize the OCR reader
    reader = easyocr.Reader(['en'], gpu=True)

    # Loop through the video frames
    while cap.isOpened():

        # Read a frame from the video
        success, frame = cap.read()
        if success:
            # Run YOLOv8 inference on the frame
            detections = model(frame, conf=0.45, verbose=False)[0]

            # Iterate over the license plate detections
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection

                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)

                # Draw license plate
                cv2.rectangle(frame, (x1, y1), (x2, y2), (50,150,250), 2)
                
                # Extract license plate crop
                licence_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]

                # Read license plate number using OCR
                ocr_detections = reader.readtext(licence_plate_crop, detail=1)

                for det in ocr_detections:
                    bbox, text, score = det

                    # License plate number post-processing
                    text = text.upper().replace(' ', '')
                    text = text.replace('*', '')
                    text = text.replace(']', '')
                    text = text.replace('}', '')

                    # Show license plate number if confidence is at least 0.75
                    if score >= 0.75:
                        cv2.putText(frame, text , (x1, y1 - 2), 0, 2 / 3, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

            # Display the results
            cv2.imshow("frame", frame)
            cv2.waitKey(1)

        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict()