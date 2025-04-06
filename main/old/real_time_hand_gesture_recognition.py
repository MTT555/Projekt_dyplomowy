import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import string
import joblib

def run_detection():
    model = tf.keras.models.load_model('models/model.h5')
    scaler = joblib.load('other/scaler.pkl')

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # classes = list(string.ascii_uppercase) + [str(i) for i in range(10)]
    classes = ['A', 'B', 'C']
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            row = []
            for lm in hand_landmarks.landmark:
                row.append(lm.x)
                row.append(lm.y)

            X_input = np.array(row).reshape(1, -1)
            X_input_scaled = scaler.transform(X_input)
            
            pred_prob = model.predict(X_input_scaled)[0]
            pred_class = np.argmax(pred_prob)
            letter = classes[pred_class]
            
            cv2.putText(frame, letter, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Detekcja znaków z obrazu z kamery", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
