import cv2
import mediapipe as mp
import csv
import os

def collect_data():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    CSV_FILE = 'data/data.csv'
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = []
            for i in range(21):
                header += [f'x{i}', f'y{i}']
            header += ['label', 'index'] 
            writer.writerow(header)

    cap = cv2.VideoCapture(0)

    current_label = None
    flip_vertical = False
    print("Wpisz literę [A-Z] lub liczbę [0-9], jaką chcesz zbierać (np. A), lub 'q' aby wyjść:")

    images_dir = '../images'
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    cv2.namedWindow("Zbieranie danych o znakach", cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if flip_vertical:
            frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        _, _, window_width, window_height = cv2.getWindowImageRect("Zbieranie danych o znakach")

        if window_width > 0 and window_height > 0:
            frame_resized = cv2.resize(frame, (window_width, window_height))
        else:
            frame_resized = frame

        
        roi = frame[10:50, 10:200]
        mean_brightness = cv2.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))[0]

        if mean_brightness > 127:
            text_color = (0, 0, 0)
        else:
            text_color = (255, 255, 255)

        cv2.putText(frame, f'Litera/Liczba: {str(current_label)}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        if 'new_index' in locals():
            index_display = str(new_index)
        else:
            index_display = 'N/A'

        cv2.putText(frame, f'Indeks znaku (ostatniego): {index_display}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        cv2.imshow("Zbieranie danych o znakach", frame)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break

        if key == 9:
            flip_vertical = not flip_vertical 
            print(f"Przerzucanie obrazu w pionie: {'Włączone' if flip_vertical else 'Wyłączone'}")
        if key == 32 or key == 13:
            if results.multi_hand_landmarks and current_label is not None:
                hand = results.multi_hand_landmarks[0]
                row = []
                for lm in hand.landmark:
                    row.append(lm.x)
                    row.append(lm.y)

                label_dir = os.path.join(images_dir, current_label)
                if not os.path.exists(label_dir):
                    os.makedirs(label_dir)

                max_index = -1
                for file_name in os.listdir(label_dir):
                    if file_name.startswith(f"{current_label}_") and file_name.endswith(".jpg"):
                        try:
                            number_part = file_name[len(f"{current_label}_"):-4]
                            num = int(number_part)
                            if num > max_index:
                                max_index = num
                        except ValueError:
                            pass

                new_index = max_index + 1

                row.append(current_label)
                row.append(str(new_index))

                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                with open(CSV_FILE, 'r') as f:
                    total_lines = sum(1 for _ in f) - 1 
                print("#" * 20)
                print(f"Zapisano {current_label} o indeksie {new_index} w pliku CSV.")
                print(f"Aktualnie w pliku CSV jest {total_lines} przykładów.")
                image_filename = f"{current_label}_{new_index}.jpg"
                image_path = os.path.join(label_dir, image_filename)
                cv2.imwrite(image_path, frame)
                print(f"Zapisano obraz w {image_path}.")

                count_files = len([fname for fname in os.listdir(label_dir)
                                if fname.startswith(f"{current_label}_") and fname.endswith(".jpg")])
                print(f"W folderze '{current_label}' jest {count_files} plików.")
                print("#" * 20)
        if key != -1 and chr(key).isalpha():
            current_label = chr(key).upper()
            print(f"Wybrano literę/cyfrę: {current_label}")

    cap.release()
    cv2.destroyAllWindows()