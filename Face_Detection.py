import cv2
import pathlib

# Load pre-trained Haar cascade classifier for face detection
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))

# Open camera for capturing video frames
cam = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture frame from camera")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

    # Display the frame with detected faces
    cv2.imshow('Face Detection', frame)
    
    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
