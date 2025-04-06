import os
import tkinter as tk
from tkinter import ttk, scrolledtext
import cv2
import numpy as np
import mediapipe as mp
import joblib
import tensorflow as tf
from PIL import Image, ImageTk
import pandas as pd
import os

def show_top10_predictions_text(app, pred_prob):
    if not hasattr(app, 'prob_window') or not app.prob_window.winfo_exists():
        app.prob_window = tk.Toplevel(app.root)
        app.prob_window.title("Najwyższe prawdopodobieństwa")
        app.prob_labels = []
        
        for i in range(10):
            lbl = tk.Label(app.prob_window, text="", font=("Roboto", 12))
            lbl.pack(anchor='w', padx=10, pady=2)
            app.prob_labels.append(lbl)
    
    top10_indices = np.argsort(pred_prob)[::-1][:10]
    top10_probs = pred_prob[top10_indices]
    top10_classes = [app.classes[i] for i in top10_indices]

    num_to_show = min(len(top10_classes), 10)
    
    for i in range(num_to_show):
        znak = top10_classes[i]
        prob = top10_probs[i] * 100
        app.prob_labels[i].config(text=f"{i+1}. {znak}: {prob:.2f}%")
    
    for i in range(num_to_show, 10):
        app.prob_labels[i].config(text="")

def run_text_detection(app):

    csv_path = app.csv_file_var.get()
    if not csv_path or not os.path.exists(csv_path):
        app.log("Brak ścieżki do pliku CSV – nie można wykryć klas.")
        return

    try:
        df = pd.read_csv(csv_path)
        if 'label' not in df.columns:
            app.log("Brak kolumny 'label' w pliku CSV – nie można wykryć klas.")
            return
        app.classes = sorted(df['label'].unique().tolist())
        app.log(f"Wykryto klasy: {app.classes}")
    except Exception as e:
        app.log(f"Błąd przy odczycie klas z CSV: {e}")
        return
    
    model = tf.keras.models.load_model(app.model_file_var.get())
    scaler = joblib.load(app.scaler_file_var.get())

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils


    cap = cv2.VideoCapture(app.current_camera_index)

    def update_text_detection():
        if not app.text_detection_running:
            cap.release()
            hands.close()
            return

        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            letter = ""

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y])

                X_input = np.array(row).reshape(1, -1)
                X_input_scaled = scaler.transform(X_input)
                pred_prob = model.predict(X_input_scaled)[0]

                pred_class = np.argmax(pred_prob)
                max_prob = pred_prob[pred_class]

                try:
                    threshold = float(app.text_threshold_var.get())
                except ValueError:
                    threshold = 0.7 

                if max_prob >= threshold:
                    letter = app.classes[pred_class]
                    if letter == '#':
                        letter = ' '

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            frame_rgb_for_tk = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb_for_tk)
            imgtk = ImageTk.PhotoImage(image=img)
            app.text_det_camera_label.imgtk = imgtk
            app.text_det_camera_label.configure(image=imgtk)

            if letter and letter != app.last_text_detected_letter:
                check_letter_correctness_and_stats(app, letter)
                show_top10_predictions_text(app, pred_prob)
                app.last_text_detected_letter = letter

        try:
            interval_ms = int(app.text_interval_var.get())
        except ValueError:
            interval_ms = 50 

        app.root.after(interval_ms, update_text_detection)

    update_text_detection()

def check_letter_correctness_and_stats(app, recognized_letter):
   
    if not app.text_file_content or app.current_char_index >= len(app.text_file_content):
        return

    expected_char = app.text_file_content[app.current_char_index]

    while expected_char in [' ', '\n', '\r'] and app.current_char_index < len(app.text_file_content):
        app.current_char_index += 1
        if app.current_char_index < len(app.text_file_content):
            expected_char = app.text_file_content[app.current_char_index]
        else:
            return 

    if recognized_letter.upper() == expected_char.upper():
       
        start_index = f"1.0+{app.current_char_index}c"
        end_index = f"1.0+{app.current_char_index+1}c"
        app.text_detection_text.tag_add("correct", start_index, end_index)
        app.current_char_index += 1
        app.recognized_signs_count += 1
    else:
        app.failed_attempts += 1

    update_text_stats(app)

def update_text_stats(app):
    remaining = app.total_sign_count - app.recognized_signs_count
    stats_msg = (
        f"Poprawnie rozpoznane: {app.recognized_signs_count} / {app.total_sign_count}\n"
        f"Błędy (nieudane próby): {app.failed_attempts}\n"
        f"Pozostało znaków: {remaining}"
    )
    app.text_stats_label.config(text=stats_msg)

def create_text_detection_tab(app):
    app.tab_text_detection = ttk.Frame(app.notebook)
    app.notebook.add(app.tab_text_detection, text="Miganie tekstu")

    main_frame = ttk.Frame(app.tab_text_detection)
    main_frame.pack(fill=tk.BOTH, expand=True)

    left_frame = ttk.Frame(main_frame)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    right_frame = ttk.Frame(main_frame, width=300)
    right_frame.pack(side=tk.RIGHT, fill=tk.Y)

    app.text_det_camera_label = ttk.Label(left_frame, text="Podgląd kamery (tekst)", font=("Roboto", 12))
    app.text_det_camera_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    app.text_detection_text = scrolledtext.ScrolledText(right_frame, height=20, wrap=tk.WORD, font=("Roboto", 12))
    app.text_detection_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    app.text_detection_text.tag_config("correct", foreground="green", font=("Roboto", 12, "bold"))

    interval_frame = ttk.Frame(right_frame)
    interval_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

    interval_label = ttk.Label(interval_frame, text="Interwał (ms):", font=("Roboto", 12))
    interval_label.pack(side=tk.LEFT, padx=5)

    app.text_interval_var = tk.StringVar(value="50")
    interval_entry = ttk.Entry(interval_frame, textvariable=app.text_interval_var, width=7, font=("Roboto", 12))
    interval_entry.pack(side=tk.LEFT, padx=5)

    threshold_label = ttk.Label(interval_frame, text="Próg:", font=("Roboto", 12))
    threshold_label.pack(side=tk.LEFT, padx=5)
    app.text_threshold_var = tk.StringVar(value="0.7")  
    threshold_entry = ttk.Entry(interval_frame, textvariable=app.text_threshold_var, width=5, font=("Roboto", 12))
    threshold_entry.pack(side=tk.LEFT, padx=5)

    file_select_frame = ttk.Frame(right_frame)
    file_select_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

    ttk.Label(file_select_frame, text="Wybierz plik z tekstem:", font=("Roboto", 12)).pack(side=tk.LEFT, padx=5)
    app.text_file_var = tk.StringVar(value="")
    app.text_file_combo = ttk.Combobox(file_select_frame, textvariable=app.text_file_var, state="readonly")
    app.text_file_combo.pack(side=tk.LEFT, padx=5)

    text_files_dir = "text_files" 
    if os.path.isdir(text_files_dir):
        text_files = [
            f for f in os.listdir(text_files_dir)
            if os.path.isfile(os.path.join(text_files_dir, f))
        ]
        app.text_file_combo['values'] = text_files
    else:
        text_files = []
        app.log(f"Folder {text_files_dir} nie istnieje!")

    btn_frame = ttk.Frame(right_frame)
    btn_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

    app.text_stats_label = ttk.Label(right_frame, text="", font=("Roboto", 12))
    app.text_stats_label.pack(side=tk.TOP, fill=tk.X, pady=5)

    def load_text_file():
        file_name = app.text_file_var.get()
        file_path = os.path.join(text_files_dir, file_name)
        if not os.path.exists(file_path):
            app.log(f"Plik {file_path} nie istnieje!")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            app.text_file_content = f.read()

        app.text_detection_text.delete('1.0', tk.END)
        app.text_detection_text.insert(tk.END, app.text_file_content)

        app.total_sign_count = sum(1 for ch in app.text_file_content if ch not in [' ', '\n', '\r'])
        app.current_char_index = 0
        app.recognized_signs_count = 0
        app.failed_attempts = 0
        update_text_stats(app)

        app.log(f"Wczytano plik tekstowy: {file_path}")

    load_button = ttk.Button(btn_frame, text="Wczytaj tekst", command=load_text_file)
    load_button.pack(side=tk.LEFT, padx=5)

    def start_text_detection():
        if app.text_detection_running:
            return
        app.text_detection_running = True
        start_text_det_button.config(state="disabled")
        stop_text_det_button.config(state="normal")

        import threading
        t = threading.Thread(target=run_text_detection, args=(app,))
        t.start()

    def stop_text_detection():
        app.text_detection_running = False
        start_text_det_button.config(state="normal")
        stop_text_det_button.config(state="disabled")

    start_text_det_button = ttk.Button(btn_frame, text="Start", command=start_text_detection)
    start_text_det_button.pack(side=tk.LEFT, padx=5)

    stop_text_det_button = ttk.Button(btn_frame, text="Stop", command=stop_text_detection, state="disabled")
    stop_text_det_button.pack(side=tk.LEFT, padx=5)

    app.text_file_content = ""
    app.current_char_index = 0
    app.text_detection_running = False
    app.last_text_detected_letter = ""

    app.total_sign_count = 0
    app.recognized_signs_count = 0
    app.failed_attempts = 0
    update_text_stats(app)