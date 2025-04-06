import cv2
import joblib
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
import tensorflow as tf
import tkinter as tk
from tkinter import ttk, scrolledtext
import pandas as pd
import os

def show_top10_predictions(app, pred_prob):
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

def run_detection(app):
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

    app.last_detected_letter = ""
    app.next_letter = ""
    
    cap = cv2.VideoCapture(app.current_camera_index)
    
    def update_detection():
        if not app.detection_running:
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
                    threshold = float(app.threshold_var.get())
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
            app.det_camera_label.imgtk = imgtk
            app.det_camera_label.configure(image=imgtk)
            
            if letter and letter != app.last_detected_letter:
                app.last_detected_letter = letter
                app.next_letter = letter

                if not app.enter_mode_var.get():
                    app.det_text.insert(tk.END, letter)
                    app.det_text.see(tk.END)
                
                show_top10_predictions(app, pred_prob)
        
        try:
            interval_ms = int(app.interval_var.get())
        except ValueError:
            interval_ms = 1000
        
        app.root.after(interval_ms, update_detection)
    
    update_detection()

def on_enter(event, app):
    if app.enter_mode_var.get() and app.next_letter:
        app.det_text.insert(tk.END, app.next_letter)
        app.det_text.see(tk.END)
        app.next_letter = ""

def create_detection_tab(app):
    app.det_main_frame = ttk.Frame(app.tab_detection)
    app.det_main_frame.pack(fill=tk.BOTH, expand=True)

    app.det_left_frame = ttk.Frame(app.det_main_frame)
    app.det_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    app.det_right_frame = ttk.Frame(app.det_main_frame, width=300)
    app.det_right_frame.pack(side=tk.RIGHT, fill=tk.Y)

   
    app.det_camera_label = ttk.Label(app.det_left_frame, font=("Roboto", 12))
    app.det_camera_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    app.det_text = scrolledtext.ScrolledText(app.det_right_frame, height=20, wrap=tk.WORD, font=("Roboto", 12))
    app.det_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    style = ttk.Style()
    style.configure("My.TButton", font=("Roboto", 10))

    control_frame = ttk.Frame(app.det_right_frame)
    control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    interval_frame = ttk.Frame(control_frame)
    interval_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

    interval_label = ttk.Label(interval_frame, text="Interwał (ms):", font=("Roboto", 12))
    interval_label.pack(side=tk.LEFT, padx=5)

    app.interval_var = tk.StringVar(value="1000")
    interval_entry = ttk.Entry(interval_frame, textvariable=app.interval_var, width=7, font=("Roboto", 12))
    interval_entry.pack(side=tk.LEFT, padx=5)

    threshold_label = ttk.Label(interval_frame, text="Próg:", font=("Roboto", 12))
    threshold_label.pack(side=tk.LEFT, padx=5)

    app.threshold_var = tk.StringVar(value="0.7")
    threshold_entry = ttk.Entry(interval_frame, textvariable=app.threshold_var, width=5, font=("Roboto", 12))
    threshold_entry.pack(side=tk.LEFT, padx=5)

    app.enter_mode_var = tk.BooleanVar(value=False)
    enter_checkbutton = ttk.Checkbutton(
        app.det_right_frame,
        text="Wstawiaj znak tylko po Enterze",
        variable=app.enter_mode_var
    )
    enter_checkbutton.pack(side=tk.TOP, anchor='w', padx=10, pady=5)

    btn_frame = ttk.Frame(control_frame)
    btn_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

    app.start_det_button = ttk.Button(btn_frame, text="Start Detekcji", command=app.start_detection, style="My.TButton")
    app.start_det_button.pack(side=tk.LEFT, padx=5)

    app.stop_det_button = ttk.Button(btn_frame, text="Stop Detekcji", command=app.stop_detection, state="disabled", style="My.TButton")
    app.stop_det_button.pack(side=tk.LEFT, padx=5)

    app.clear_screen_button = ttk.Button(
        btn_frame,
        text="Wyczyść ekran",
        command=lambda: app.det_text.delete('1.0', tk.END),
        style="My.TButton"
    )
    app.clear_screen_button.pack(side=tk.LEFT, padx=5)

    app.next_letter = "" 
    app.root.bind_all("<Return>", lambda event: on_enter(event, app))

