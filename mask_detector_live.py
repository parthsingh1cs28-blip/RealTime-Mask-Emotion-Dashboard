import cv2
import numpy as np
from tensorflow.keras.models import load_model
model = load_model("models/mask_detector_model.h5")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
img_size = 64
from collections import deque
prediction_buffer = deque(maxlen=5) # was 10 then 15 
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 8)

    for (x, y, w, h) in faces:
        face = frame[y+10:y+h-10, x+10:x+w-10]
        face = cv2.resize(face, (img_size, img_size))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype("float32") / 255.0
        face = np.reshape(face, (1, img_size, img_size, 3))
        prediction = model.predict(face, verbose=0)[0][0]

        prediction_buffer.append(prediction)

        avg_prediction = sum(prediction_buffer) / len(prediction_buffer)
        if avg_prediction < 0.5:
            label = "With Mask"
            confidence = (1 - avg_prediction) * 100
            color = (0, 255, 0)
        else:
            label = "Without Mask"
            confidence = avg_prediction * 100
            color = (0, 0, 255)
        #prediction = model.predict(face, verbose=0)#[0][0]
        #if prediction[0][0] < 0.5:
        #    label = "With Mask"
        #    confidence = (1 - prediction) * 100
        #    color = (0, 255, 0)
        #else:
        #    label = "Without Mask"
        #    confidence = prediction * 100
        #    color = (0, 0, 255)
        #print(prediction[0][0])
        #text = f"{label} ({confidence:.2f}%)"
        #cv2.putText(frame, text, (x, y-10),
         #           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        #cv2.putText(frame, label, (x, y-10),
        text = f"{label} ({confidence:.2f}%)"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        bar_width = 150
        bar_height = 15
        bar_x = x
        bar_y = y + h + 10
        filled_width = int((confidence / 100) * bar_width)

        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_width, bar_y + bar_height),
                      (200, 200, 200), 2)

        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + filled_width, bar_y + bar_height),
                      color, -1)
    cv2.imshow("Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()