import cv2
import numpy as np
from keras.models import load_model

# ==========================
# Load Trained Model
# ==========================
model = load_model("face_mask_detector_fixed.keras")

# ==========================
# Load Face Detector
# ==========================
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==========================
# Open Webcam
# ==========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

# ==========================
# Webcam Loop
# ==========================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # Loop through all detected faces
    for (x, y, w, h) in faces:

        # Crop face
        face = frame[y:y+h, x:x+w]

        # Resize to model input size
        face = cv2.resize(face, (128, 128))

        # Normalize
        face = face.astype("float32") / 255.0

        # Add batch dimension
        face = np.expand_dims(face, axis=0)

        # Prediction
        prediction = model.predict(face, verbose=0)

        # Predicted class
        label = np.argmax(prediction)

        # Confidence
        confidence = np.max(prediction)

        # Label text
        if label == 0:
            text = "Mask"
            color = (0, 255, 0)
        else:
            text = "No Mask"
            color = (0, 0, 255)

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Draw prediction text
        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # Show webcam
    cv2.imshow("Face Mask Detection", frame)

    # Exit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================
# Cleanup
# ==========================
cap.release()
cv2.destroyAllWindows()