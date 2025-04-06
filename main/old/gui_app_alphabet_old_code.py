import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import csv
import os
import numpy as np
import threading
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import sys

def disable_space_activation(widget):
    widget.bind("<space>", lambda e: "break")

class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
    def flush(self):
        pass

class EpochProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, progress_var, app_log_func, root):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_var = progress_var
        self.app_log_func = app_log_func
        self.root = root
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_epoch = epoch + 1
        progress_percent = 100.0 * current_epoch / self.total_epochs
        self.root.after(0, lambda: self.progress_var.set(progress_percent))
        msg = f"Epoch {current_epoch}/{self.total_epochs} - loss: {logs.get('loss', 0):.4f}, accuracy: {logs.get('accuracy', 0):.4f}, val_loss: {logs.get('val_loss', 0):.4f}, val_accuracy: {logs.get('val_accuracy', 0):.4f}"
        self.root.after(0, lambda: self.app_log_func(msg))

class HandDataCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikacja - Zbieranie danych, Trening i Detekcja")
        self.logs_dir = "other"
        self.logs_file_path = os.path.join(self.logs_dir, "logs.log")
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir, exist_ok=True)
        self.log_file = open(self.logs_file_path, mode='w', encoding='utf-8')
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.tab_collect = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_collect, text="Zbieranie danych")
        self.tab_train = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_train, text="Trening modelu")
        self.tab_detection = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_detection, text="Detekcja znaków")
        self.tab_instructions = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_instructions, text="Instrukcja")
        self.log_console = scrolledtext.ScrolledText(self.main_frame, height=10, wrap=tk.WORD)
        self.log_console.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)
        self.csv_file_var = tk.StringVar(value='data/data.csv')
        self.model_file_var = tk.StringVar(value='models/model.h5')
        self.scaler_file_var = tk.StringVar(value='other/scaler.pkl')
        self.images_dir = 'images'
        self._prepare_directories_and_csv()
        self.current_label = None
        self.flip_vertical = False
        self.new_index = None
        self.default_static_image_mode = False
        self.default_max_num_hands = 1
        self.default_model_complexity = 1
        self.default_min_detection_confidence = 0.5
        self.default_min_tracking_confidence = 0.5
        self.static_image_mode_var = tk.BooleanVar(value=self.default_static_image_mode)
        self.max_num_hands_var = tk.IntVar(value=self.default_max_num_hands)
        self.model_complexity_var = tk.IntVar(value=self.default_model_complexity)
        self.min_detection_confidence_var = tk.IntVar(value=int(self.default_min_detection_confidence * 100))
        self.min_tracking_confidence_var = tk.IntVar(value=int(self.default_min_tracking_confidence * 100))
        self.hands = None
        self._init_mediapipe_hands()
        self.available_cameras = self._detect_cameras(max_cameras=5)
        if not self.available_cameras:
            raise RuntimeError("Nie wykryto żadnej dostępnej kamery w systemie!")
        self.current_camera_index = self.available_cameras[0]
        self.cap = cv2.VideoCapture(self.current_camera_index)
        self._create_tab_collect()
        self._create_tab_train()
        self._create_tab_detection()
        self._create_tab_instructions()
        self.root.bind_all("<space>", self._on_space_or_enter)
        self.root.bind_all("<Return>", self._on_space_or_enter)
        self.update_frame()
        self.detection_running = False
    def reset_to_defaults(self):
        self.brightness_var.set(0)
        self.contrast_var.set(100)
        self.gamma_var.set(100)
        self.r_var.set(0)
        self.g_var.set(0)
        self.b_var.set(0)
        self.static_image_mode_var.set(self.default_static_image_mode)
        self.max_num_hands_var.set(self.default_max_num_hands)
        self.model_complexity_var.set(self.default_model_complexity)
        self.min_detection_confidence_var.set(int(self.default_min_detection_confidence * 100))
        self.min_tracking_confidence_var.set(int(self.default_min_tracking_confidence * 100))
        self.update_scale_label(self.brightness_var, self.brightness_value_label)
        self.update_scale_label(self.contrast_var, self.contrast_value_label)
        self.update_scale_label(self.gamma_var, self.gamma_value_label)
        self.update_scale_label(self.r_var, self.r_value_label)
        self.update_scale_label(self.g_var, self.g_value_label)
        self.update_scale_label(self.b_var, self.b_value_label)
        self.update_scale_label(self.max_num_hands_var, self.max_hands_label)
        self.update_scale_label(self.model_complexity_var, self.model_complexity_label)
        self.update_scale_label(self.min_detection_confidence_var, self.min_detection_label)
        self.update_scale_label(self.min_tracking_confidence_var, self.min_tracking_label)
        self._init_mediapipe_hands()
        self.log("Przywrócono wszystkie ustawienia do wartości domyślnych.")
    def update_mediapipe_settings(self):
        self._init_mediapipe_hands()
        self.log("Zaktualizowano ustawienia MediaPipe Hands.")
    def start_training(self):
        self.progress_var.set(0.0)
        training_thread = threading.Thread(target=self.run_training_in_thread)
        training_thread.start()
    def run_training_in_thread(self):
        csv_path = self.csv_file_var.get()
        model_path = self.model_file_var.get()
        scaler_path = self.scaler_file_var.get()
        test_size = self.test_size_var.get()
        random_state = self.random_state_var.get()
        epochs = self.epochs_var.get()
        batch_size = self.batch_size_var.get()
        patience = self.patience_var.get()
        if not os.path.exists(csv_path):
            self.log(f"Plik CSV {csv_path} nie istnieje – brak danych do trenowania!")
            return
        df = pd.read_csv(csv_path)
        if 'index' in df.columns:
            df.drop(columns=['index'], inplace=True)
        if 'label' not in df.columns:
            self.log("Brak kolumny 'label' w CSV! Nie można trenować.")
            return
        X = df.drop('label', axis=1)
        y = df['label']
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError as e:
            self.log(f"Błąd przy dzieleniu zbioru: {e}")
            return
        y_train_encoded = pd.get_dummies(y_train)
        y_test_encoded = pd.get_dummies(y_test)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        if not os.path.exists(os.path.dirname(scaler_path)):
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        self.log(f"Zapisano scaler do {scaler_path}")
        num_features = X_train_scaled.shape[1]
        num_classes = y_train_encoded.shape[1]
        model = keras.Sequential([
            layers.Input(shape=(num_features,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.log(f"Model summary:\n{model.summary()}")
        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        progress_callback = EpochProgressCallback(
            total_epochs=epochs,
            progress_var=self.progress_var,
            app_log_func=self.log,
            root=self.root
        )
        history = model.fit(
            X_train_scaled, y_train_encoded,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, progress_callback],
            verbose=0
        )
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        self.log(f"Zapisano model w {model_path}")
        test_loss, test_acc = model.evaluate(X_test_scaled, y_test_encoded, verbose=0)
        self.log(f"Test accuracy: {test_acc:.4f}")
        from sklearn.metrics import confusion_matrix, classification_report
        y_pred_prob = model.predict(X_test_scaled)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test_encoded.values, axis=1)
        cm = confusion_matrix(y_true, y_pred)
        self.log(f"Confusion matrix:\n{cm}")
        self.log(classification_report(y_true, y_pred))
        self.log("Trening zakończony.")
        self.root.after(0, lambda: self.progress_var.set(100.0))

    def _prepare_directories_and_csv(self):
        csv_path = self.csv_file_var.get()
        if not os.path.exists(os.path.dirname(csv_path)):
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        if not os.path.exists(csv_path):
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                header = []
                for i in range(21):
                    header += [f'x{i}', f'y{i}']
                header += ['label', 'index']
                writer.writerow(header)

    def log(self, text):
        self.log_console.insert(tk.END, text + "\n")
        self.log_console.see(tk.END)
        print(text)
        if self.log_file:
            self.log_file.write(text + "\n")
            self.log_file.flush()

    def _init_mediapipe_hands(self):
        if self.hands is not None:
            self.hands.close()
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=self.static_image_mode_var.get(),
            max_num_hands=self.max_num_hands_var.get(),
            model_complexity=self.model_complexity_var.get(),
            min_detection_confidence=float(self.min_detection_confidence_var.get()) / 100.0,
            min_tracking_confidence=float(self.min_tracking_confidence_var.get()) / 100.0
        )

    def _detect_cameras(self, max_cameras=5):
        cameras = []
        for i in range(max_cameras):
            temp_cap = cv2.VideoCapture(i)
            if temp_cap.isOpened():
                ret, _ = temp_cap.read()
                if ret:
                    cameras.append(i)
                temp_cap.release()
        return cameras

    def _create_tab_collect(self):
        self.collect_left_frame = ttk.Frame(self.tab_collect)
        self.collect_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.collect_right_frame = ttk.Frame(self.tab_collect, width=550)
        self.collect_right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.camera_label = ttk.Label(self.collect_left_frame)
        self.camera_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.controls_frame = ttk.Frame(self.collect_right_frame)
        self.controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.mp_frame = ttk.Frame(self.collect_right_frame)
        self.mp_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        self._create_controls_frame()
        self._create_mediapipe_frame()

    def _create_controls_frame(self):
        ttk.Label(self.controls_frame, text="Wybierz kamerę:").pack(pady=(5,0))
        self.camera_combo = ttk.Combobox(self.controls_frame, state="readonly", values=self.available_cameras)
        self.camera_combo.set(self.current_camera_index)
        self.camera_combo.pack(padx=5, pady=5)
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_select)
        ttk.Label(self.controls_frame, text="Podaj literę/cyfrę do zbierania:").pack(pady=5)
        self.label_entry = ttk.Entry(self.controls_frame)
        self.label_entry.pack(padx=5, pady=5)
        set_label_button = ttk.Button(self.controls_frame, text="Ustaw literę", command=self.set_label)
        set_label_button.pack(pady=5)
        disable_space_activation(set_label_button)
        save_button = ttk.Button(self.controls_frame, text="Zapisz dane", command=self.save_data)
        save_button.pack(pady=5)
        disable_space_activation(save_button)
        flip_button = ttk.Button(self.controls_frame, text="Flip w pionie (tab)", command=self.toggle_flip)
        flip_button.pack(pady=5)
        disable_space_activation(flip_button)
        ttk.Label(self.controls_frame, text="--- Zarządzanie danymi ---").pack(pady=(10,0))
        clear_images_button = ttk.Button(self.controls_frame, text="Wyczyść folder images", command=self.clear_images)
        clear_images_button.pack(pady=5)
        disable_space_activation(clear_images_button)
        clear_csv_button = ttk.Button(self.controls_frame, text="Wyzeruj plik CSV", command=self.clear_csv)
        clear_csv_button.pack(pady=5)
        disable_space_activation(clear_csv_button)
        ttk.Label(self.controls_frame, text="--- Reset ustawień ---").pack(pady=(10,0))
        reset_button = ttk.Button(self.controls_frame, text="Przywróć domyślne", command=self.reset_to_defaults)
        reset_button.pack(pady=5)
        disable_space_activation(reset_button)
        quit_button = ttk.Button(self.controls_frame, text="Wyjdź (q)", command=self.quit_app)
        quit_button.pack(pady=5)
        disable_space_activation(quit_button)
        ttk.Label(self.controls_frame, text="--- Ustawienia obrazu ---").pack(pady=(10,5))
        ttk.Label(self.controls_frame, text="Jasność (beta):").pack()
        self.brightness_var = tk.IntVar(value=0)
        self.brightness_value_label = ttk.Label(self.controls_frame, text="0")
        self.brightness_value_label.pack()
        self.brightness_scale = ttk.Scale(self.controls_frame, from_=-100, to=100,
                                          variable=self.brightness_var, orient=tk.HORIZONTAL,
                                          command=lambda e: self.update_scale_label(self.brightness_var, self.brightness_value_label))
        self.brightness_scale.pack(padx=5)
        ttk.Label(self.controls_frame, text="Kontrast (alpha%):").pack()
        self.contrast_var = tk.IntVar(value=100)
        self.contrast_value_label = ttk.Label(self.controls_frame, text="100")
        self.contrast_value_label.pack()
        self.contrast_scale = ttk.Scale(self.controls_frame, from_=0, to=200,
                                        variable=self.contrast_var, orient=tk.HORIZONTAL,
                                        command=lambda e: self.update_scale_label(self.contrast_var, self.contrast_value_label))
        self.contrast_scale.pack(padx=5)
        ttk.Label(self.controls_frame, text="Gamma (1.0 = brak):").pack()
        self.gamma_var = tk.IntVar(value=100)
        self.gamma_value_label = ttk.Label(self.controls_frame, text="100")
        self.gamma_value_label.pack()
        self.gamma_scale = ttk.Scale(self.controls_frame, from_=1, to=300,
                                     variable=self.gamma_var, orient=tk.HORIZONTAL,
                                     command=lambda e: self.update_scale_label(self.gamma_var, self.gamma_value_label))
        self.gamma_scale.pack(padx=5)
        ttk.Label(self.controls_frame, text="Przesunięcie koloru (R, G, B):").pack()
        frame_rgb = ttk.Frame(self.controls_frame)
        frame_rgb.pack(padx=5, pady=5, fill=tk.X)
        ttk.Label(frame_rgb, text="R:").grid(row=0, column=0, sticky=tk.W)
        self.r_var = tk.IntVar(value=0)
        self.r_value_label = ttk.Label(frame_rgb, text="0")
        self.r_value_label.grid(row=0, column=2, sticky=tk.W, padx=(5,0))
        r_scale = ttk.Scale(frame_rgb, from_=-50, to=50,
                            variable=self.r_var, orient=tk.HORIZONTAL, length=120,
                            command=lambda e: self.update_scale_label(self.r_var, self.r_value_label))
        r_scale.grid(row=0, column=1, padx=5)
        ttk.Label(frame_rgb, text="G:").grid(row=1, column=0, sticky=tk.W)
        self.g_var = tk.IntVar(value=0)
        self.g_value_label = ttk.Label(frame_rgb, text="0")
        self.g_value_label.grid(row=1, column=2, sticky=tk.W, padx=(5,0))
        g_scale = ttk.Scale(frame_rgb, from_=-50, to=50,
                            variable=self.g_var, orient=tk.HORIZONTAL, length=120,
                            command=lambda e: self.update_scale_label(self.g_var, self.g_value_label))
        g_scale.grid(row=1, column=1, padx=5)
        ttk.Label(frame_rgb, text="B:").grid(row=2, column=0, sticky=tk.W)
        self.b_var = tk.IntVar(value=0)
        self.b_value_label = ttk.Label(frame_rgb, text="0")
        self.b_value_label.grid(row=2, column=2, sticky=tk.W, padx=(5,0))
        b_scale = ttk.Scale(frame_rgb, from_=-50, to=50,
                            variable=self.b_var, orient=tk.HORIZONTAL, length=120,
                            command=lambda e: self.update_scale_label(self.b_var, self.b_value_label))
        b_scale.grid(row=2, column=1, padx=5)

    def _create_mediapipe_frame(self):
        ttk.Label(self.mp_frame, text="--- Ustawienia MediaPipe ---").pack(pady=(10,5))
        self.static_check = ttk.Checkbutton(self.mp_frame,
                                            text="static_image_mode (True = zdjęcia statyczne)",
                                            variable=self.static_image_mode_var)
        self.static_check.pack(padx=5, pady=2)
        ttk.Label(self.mp_frame, text="max_num_hands:").pack()
        self.max_hands_label = ttk.Label(self.mp_frame, text=str(self.max_num_hands_var.get()))
        self.max_hands_label.pack()
        self.max_num_hands_scale = ttk.Scale(self.mp_frame, from_=1, to=4,
                                             variable=self.max_num_hands_var, orient=tk.HORIZONTAL,
                                             command=lambda e: self.update_scale_label(self.max_num_hands_var, self.max_hands_label))
        self.max_num_hands_scale.pack(padx=5, pady=(0,5))
        ttk.Label(self.mp_frame, text="model_complexity (0-2):").pack()
        self.model_complexity_label = ttk.Label(self.mp_frame, text=str(self.model_complexity_var.get()))
        self.model_complexity_label.pack()
        self.model_complexity_scale = ttk.Scale(self.mp_frame, from_=0, to=2,
                                                variable=self.model_complexity_var, orient=tk.HORIZONTAL,
                                                command=lambda e: self.update_scale_label(self.model_complexity_var, self.model_complexity_label))
        self.model_complexity_scale.pack(padx=5, pady=(0,5))
        ttk.Label(self.mp_frame, text="min_detection_confidence (%):").pack()
        self.min_detection_label = ttk.Label(self.mp_frame, text=str(self.min_detection_confidence_var.get()))
        self.min_detection_label.pack()
        self.min_detection_scale = ttk.Scale(self.mp_frame, from_=0, to=100,
                                             variable=self.min_detection_confidence_var, orient=tk.HORIZONTAL,
                                             command=lambda e: self.update_scale_label(self.min_detection_confidence_var, self.min_detection_label))
        self.min_detection_scale.pack(padx=5, pady=(0,5))
        ttk.Label(self.mp_frame, text="min_tracking_confidence (%):").pack()
        self.min_tracking_label = ttk.Label(self.mp_frame, text=str(self.min_tracking_confidence_var.get()))
        self.min_tracking_label.pack()
        self.min_tracking_scale = ttk.Scale(self.mp_frame, from_=0, to=100,
                                            variable=self.min_tracking_confidence_var, orient=tk.HORIZONTAL,
                                            command=lambda e: self.update_scale_label(self.min_tracking_confidence_var, self.min_tracking_label))
        self.min_tracking_scale.pack(padx=5, pady=(0,5))
        apply_mediapipe_button = ttk.Button(self.mp_frame,
                                            text="Zastosuj zmiany w MediaPipe",
                                            command=self.update_mediapipe_settings)
        apply_mediapipe_button.pack(padx=5, pady=5)
        disable_space_activation(apply_mediapipe_button)
        file_label_frame = ttk.LabelFrame(self.mp_frame, text="Nazwa plików (domyślne wartości)")
        file_label_frame.pack(fill=tk.X, pady=(10,0))
        ttk.Label(file_label_frame, text="CSV (wej./wyj.):").grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        csv_entry = ttk.Entry(file_label_frame, textvariable=self.csv_file_var, width=30)
        csv_entry.grid(row=0, column=1, sticky=tk.W, padx=2, pady=2)
        ttk.Label(file_label_frame, text="Model (wyj.):").grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        model_entry = ttk.Entry(file_label_frame, textvariable=self.model_file_var, width=30)
        model_entry.grid(row=1, column=1, sticky=tk.W, padx=2, pady=2)
        ttk.Label(file_label_frame, text="Scaler (wyj.):").grid(row=2, column=0, sticky=tk.W, padx=2, pady=2)
        scaler_entry = ttk.Entry(file_label_frame, textvariable=self.scaler_file_var, width=30)
        scaler_entry.grid(row=2, column=1, sticky=tk.W, padx=2, pady=2)

    def _create_tab_train(self):
        self.train_main_frame = ttk.Frame(self.tab_train)
        self.train_main_frame.pack(fill=tk.BOTH, expand=True)
        config_frame = ttk.LabelFrame(self.train_main_frame, text="Konfiguracja treningu")
        config_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        ttk.Label(config_frame, text="Test size (np. 0.2):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.test_size_var = tk.DoubleVar(value=0.2)
        ttk.Entry(config_frame, textvariable=self.test_size_var, width=8).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(config_frame, text="Random state:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.random_state_var = tk.IntVar(value=42)
        ttk.Entry(config_frame, textvariable=self.random_state_var, width=8).grid(row=1, column=1, sticky=tk.W)
        ttk.Label(config_frame, text="Epochs:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.epochs_var = tk.IntVar(value=50)
        ttk.Entry(config_frame, textvariable=self.epochs_var, width=8).grid(row=2, column=1, sticky=tk.W)
        ttk.Label(config_frame, text="Batch size:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.batch_size_var = tk.IntVar(value=32)
        ttk.Entry(config_frame, textvariable=self.batch_size_var, width=8).grid(row=3, column=1, sticky=tk.W)
        ttk.Label(config_frame, text="Patience (EarlyStopping):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.patience_var = tk.IntVar(value=5)
        ttk.Entry(config_frame, textvariable=self.patience_var, width=8).grid(row=4, column=1, sticky=tk.W)
        train_button_frame = ttk.Frame(self.train_main_frame)
        train_button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        start_train_button = ttk.Button(train_button_frame, text="Rozpocznij trening", command=self.start_training)
        start_train_button.pack(side=tk.LEFT, padx=5)
        disable_space_activation(start_train_button)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(self.train_main_frame, variable=self.progress_var,
                                            orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(side=tk.TOP, padx=10, pady=5)

    def _create_tab_detection(self):
        self.det_main_frame = ttk.Frame(self.tab_detection)
        self.det_main_frame.pack(fill=tk.BOTH, expand=True)
        self.det_left_frame = ttk.Frame(self.det_main_frame)
        self.det_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.det_right_frame = ttk.Frame(self.det_main_frame, width=300)
        self.det_right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.det_camera_label = ttk.Label(self.det_left_frame)
        self.det_camera_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.det_text = scrolledtext.ScrolledText(self.det_right_frame, height=20, wrap=tk.WORD)
        self.det_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        btn_frame = ttk.Frame(self.det_right_frame)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        self.start_det_button = ttk.Button(btn_frame, text="Start Detekcji", command=self.start_detection)
        self.start_det_button.pack(side=tk.LEFT, padx=5)
        self.stop_det_button = ttk.Button(btn_frame, text="Stop Detekcji", command=self.stop_detection, state="disabled")
        self.stop_det_button.pack(side=tk.LEFT, padx=5)

    def _create_tab_instructions(self):
        instructions_text = ("Instrukcja obsługi aplikacji:\n\n"
                             "1) Zakładka 'Zbieranie danych':\n"
                             "   - Wybierz kamerę, ustaw literę/cyfrę i zbieraj dane (punkty dłoni).\n"
                             "   - Spacja/Enter lub przycisk 'Zapisz dane' zapisuje dane do CSV oraz obraz do folderu images/.\n"
                             "   - Ustawienia obrazu i MediaPipe wpływają na jakość detekcji.\n"
                             "   - 'Wyczyść folder images' usuwa wszystkie zebrane obrazy, 'Wyzeruj plik CSV' resetuje dane.\n"
                             "   - 'Przywróć domyślne' – resetuje wszystkie ustawienia.\n\n"
                             "2) Zakładka 'Trening modelu':\n"
                             "   - Ustaw plik CSV, model, scaler oraz parametry treningu.\n"
                             "   - Po kliknięciu 'Rozpocznij trening' trening uruchamia się w tle, a postęp epok jest aktualizowany.\n\n"
                             "3) Zakładka 'Detekcja znaków':\n"
                             "   - Po kliknięciu 'Start Detekcji' uruchamia się moduł rozpoznawania znaków z kamery.\n"
                             "   - Obraz z kamery jest wyświetlany, a rozpoznane znaki pojawiają się w polu tekstowym.\n"
                             "   - Przycisk 'Stop Detekcji' zatrzymuje detekcję.\n\n"
                             "4) Zakładka 'Instrukcja':\n"
                             "   - Znajdziesz tutaj pełen opis działania aplikacji.\n\n"
                             "Na dole aplikacji znajduje się wspólne pole logów.\n"
                             "Miłego korzystania!\n")
        label_instruct = ttk.Label(self.tab_instructions, text=instructions_text, justify=tk.LEFT)
        label_instruct.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=10)

    def start_detection(self):
        if self.detection_running:
            return
        self.detection_running = True
        self.start_det_button.config(state="disabled")
        self.stop_det_button.config(state="normal")
        self.det_thread = threading.Thread(target=self.run_detection)
        self.det_thread.start()

    def stop_detection(self):
        self.detection_running = False
        self.start_det_button.config(state="normal")
        self.stop_det_button.config(state="disabled")

    def run_detection(self):
        model = tf.keras.models.load_model(self.model_file_var.get())
        scaler = joblib.load(self.scaler_file_var.get())
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        mp_drawing = mp.solutions.drawing_utils
        classes = ['A', 'B', 'C']
        cap = cv2.VideoCapture(self.current_camera_index)
        while self.detection_running:
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            letter = ""
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
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            frame_rgb_for_tk = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb_for_tk)
            imgtk = ImageTk.PhotoImage(image=img)
            self.det_camera_label.imgtk = imgtk
            self.det_camera_label.configure(image=imgtk)
            if letter:
                self.det_text.insert(tk.END, letter + "\n")
                self.det_text.see(tk.END)
        cap.release()
        hands.close()

    def on_camera_select(self, event=None):
        new_index = int(self.camera_combo.get())
        if new_index != self.current_camera_index:
            self.log(f"Zmieniam kamerę z {self.current_camera_index} na {new_index}...")
            self.cap.release()
            self.cap = cv2.VideoCapture(new_index)
            self.current_camera_index = new_index

    def set_label(self):
        label_text = self.label_entry.get().strip()
        if label_text:
            self.current_label = label_text.upper()
            self.log(f"Wybrano literę/cyfrę: {self.current_label}")
        else:
            self.log("Nie podano żadnej litery/cyfry!")

    def toggle_flip(self):
        self.flip_vertical = not self.flip_vertical
        status = "Włączone" if self.flip_vertical else "Wyłączone"
        self.log(f"Przerzucanie obrazu w pionie: {status}")

    def _on_space_or_enter(self, event):
        current_tab_index = self.notebook.index(self.notebook.select())
        if current_tab_index == 0:
            self.save_data()

    def save_data(self):
        if self.current_label is None:
            self.log("Najpierw ustaw literę (etykietę)!")
            return
        if not hasattr(self, 'last_frame'):
            self.log("Brak danych z kamery (jeszcze nie przetworzono klatki)!")
            return
        frame_rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            row = []
            for lm in hand.landmark:
                row.append(lm.x)
                row.append(lm.y)
            label_dir = os.path.join(self.images_dir, self.current_label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            max_index = -1
            for file_name in os.listdir(label_dir):
                if file_name.startswith(f"{self.current_label}_") and file_name.endswith(".jpg"):
                    try:
                        number_part = file_name[len(f"{self.current_label}_"):-4]
                        num = int(number_part)
                        if num > max_index:
                            max_index = num
                    except ValueError:
                        pass
            new_index = max_index + 1
            self.new_index = new_index
            row.append(self.current_label)
            row.append(str(new_index))
            csv_path = self.csv_file_var.get()
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            with open(csv_path, 'r') as f:
                total_lines = sum(1 for _ in f) - 1
            self.log("#" * 20)
            self.log(f"Zapisano {self.current_label} o indeksie {new_index} w pliku CSV: {csv_path}.")
            self.log(f"Aktualnie w pliku CSV jest {total_lines} przykładów.")
            image_filename = f"{self.current_label}_{new_index}.jpg"
            image_path = os.path.join(label_dir, image_filename)
            cv2.imwrite(image_path, self.last_frame)
            self.log(f"Zapisano obraz w {image_path}.")
            count_files = len([fname for fname in os.listdir(label_dir)
                               if fname.startswith(f"{self.current_label}_") and fname.endswith(".jpg")])
            self.log(f"W folderze '{self.current_label}' jest {count_files} plików.")
            self.log("#" * 20)
        else:
            self.log("Nie wykryto dłoni – nie można zapisać danych.")

    def clear_images(self):
        if not os.path.exists(self.images_dir):
            self.log("Folder images już jest pusty lub nie istnieje.")
            return
        confirm = messagebox.askyesno("Potwierdzenie", "Na pewno usunąć cały katalog 'images' wraz z podkatalogami?")
        if confirm:
            import shutil
            shutil.rmtree(self.images_dir)
            os.makedirs(self.images_dir)
            self.log("Wyczyszczono katalog images/ wraz z podkatalogami.")
        else:
            self.log("Anulowano czyszczenie katalogu images/.")

    def clear_csv(self):
        csv_path = self.csv_file_var.get()
        if not os.path.exists(csv_path):
            self.log(f"Plik {csv_path} nie istnieje – nie ma czego czyścić.")
            return
        confirm = messagebox.askyesno("Potwierdzenie", f"Na pewno wyzerować zawartość pliku {csv_path}?")
        if confirm:
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                header = []
                for i in range(21):
                    header += [f'x{i}', f'y{i}']
                header += ['label', 'index']
                writer.writerow(header)
            self.log(f"Wyzerowano plik CSV {csv_path} (zapisano tylko nagłówek).")
        else:
            self.log("Anulowano czyszczenie pliku CSV.")

    def update_frame(self):
        current_tab_index = self.notebook.index(self.notebook.select())
        if current_tab_index == 0 and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.flip_vertical:
                    frame = cv2.flip(frame, 1)
                frame = self.apply_image_adjustments(frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                if self.current_label is not None:
                    cv2.putText(frame, f'Litera/Liczba: {self.current_label}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                index_display = str(self.new_index) if self.new_index is not None else 'N/A'
                cv2.putText(frame, f'Indeks znaku (ostatniego): {index_display}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                self.last_frame = frame.copy()
                frame_rgb_for_tk = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb_for_tk)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
        self.root.after(20, self.update_frame)

    def apply_image_adjustments(self, frame):
        frame = frame.astype(np.float32)
        beta = float(self.brightness_var.get())
        alpha_percent = float(self.contrast_var.get())
        alpha = max(0.01, alpha_percent / 100.0)
        frame = frame * alpha + beta
        shift_r = float(self.r_var.get())
        shift_g = float(self.g_var.get())
        shift_b = float(self.b_var.get())
        frame[:, :, 2] += shift_r
        frame[:, :, 1] += shift_g
        frame[:, :, 0] += shift_b
        frame = np.clip(frame, 0, 255)
        frame = frame / 255.0
        real_gamma = max(0.01, float(self.gamma_var.get()) / 100.0)
        frame = np.power(frame, 1.0 / real_gamma)
        frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def quit_app(self):
        self.log("Zamykanie aplikacji...")
        if self.cap:
            self.cap.release()
        if self.log_file:
            self.log_file.close()
        self.root.destroy()

    def on_camera_select(self, event=None):
        new_index = int(self.camera_combo.get())
        if new_index != self.current_camera_index:
            self.log(f"Zmieniam kamerę z {self.current_camera_index} na {new_index}...")
            self.cap.release()
            self.cap = cv2.VideoCapture(new_index)
            self.current_camera_index = new_index

    def set_label(self):
        label_text = self.label_entry.get().strip()
        if label_text:
            self.current_label = label_text.upper()
            self.log(f"Wybrano literę/cyfrę: {self.current_label}")
        else:
            self.log("Nie podano żadnej litery/cyfry!")

    def toggle_flip(self):
        self.flip_vertical = not self.flip_vertical
        status = "Włączone" if self.flip_vertical else "Wyłączone"
        self.log(f"Przerzucanie obrazu w pionie: {status}")

    def _on_space_or_enter(self, event):
        current_tab_index = self.notebook.index(self.notebook.select())
        if current_tab_index == 0:
            self.save_data()

    def save_data(self):
        if self.current_label is None:
            self.log("Najpierw ustaw literę (etykietę)!")
            return
        if not hasattr(self, 'last_frame'):
            self.log("Brak danych z kamery (jeszcze nie przetworzono klatki)!")
            return
        frame_rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            row = []
            for lm in hand.landmark:
                row.append(lm.x)
                row.append(lm.y)
            label_dir = os.path.join(self.images_dir, self.current_label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            max_index = -1
            for file_name in os.listdir(label_dir):
                if file_name.startswith(f"{self.current_label}_") and file_name.endswith(".jpg"):
                    try:
                        number_part = file_name[len(f"{self.current_label}_"):-4]
                        num = int(number_part)
                        if num > max_index:
                            max_index = num
                    except ValueError:
                        pass
            new_index = max_index + 1
            self.new_index = new_index
            row.append(self.current_label)
            row.append(str(new_index))
            csv_path = self.csv_file_var.get()
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            with open(csv_path, 'r') as f:
                total_lines = sum(1 for _ in f) - 1
            self.log("#" * 20)
            self.log(f"Zapisano {self.current_label} o indeksie {new_index} w pliku CSV: {csv_path}.")
            self.log(f"Aktualnie w pliku CSV jest {total_lines} przykładów.")
            image_filename = f"{self.current_label}_{new_index}.jpg"
            image_path = os.path.join(label_dir, image_filename)
            cv2.imwrite(image_path, self.last_frame)
            self.log(f"Zapisano obraz w {image_path}.")
            count_files = len([fname for fname in os.listdir(label_dir)
                               if fname.startswith(f"{self.current_label}_") and fname.endswith(".jpg")])
            self.log(f"W folderze '{self.current_label}' jest {count_files} plików.")
            self.log("#" * 20)
        else:
            self.log("Nie wykryto dłoni – nie można zapisać danych.")

    def clear_images(self):
        if not os.path.exists(self.images_dir):
            self.log("Folder images już jest pusty lub nie istnieje.")
            return
        confirm = messagebox.askyesno("Potwierdzenie", "Na pewno usunąć cały katalog 'images' wraz z podkatalogami?")
        if confirm:
            import shutil
            shutil.rmtree(self.images_dir)
            os.makedirs(self.images_dir)
            self.log("Wyczyszczono katalog images/ wraz z podkatalogami.")
        else:
            self.log("Anulowano czyszczenie katalogu images/.")

    def clear_csv(self):
        csv_path = self.csv_file_var.get()
        if not os.path.exists(csv_path):
            self.log(f"Plik {csv_path} nie istnieje – nie ma czego czyścić.")
            return
        confirm = messagebox.askyesno("Potwierdzenie", f"Na pewno wyzerować zawartość pliku {csv_path}?")
        if confirm:
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                header = []
                for i in range(21):
                    header += [f'x{i}', f'y{i}']
                header += ['label', 'index']
                writer.writerow(header)
            self.log(f"Wyzerowano plik CSV {csv_path} (zapisano tylko nagłówek).")
        else:
            self.log("Anulowano czyszczenie pliku CSV.")

    def _create_tab_instructions(self):
        instructions_text = ("Instrukcja obsługi aplikacji:\n\n"
                             "1) Zakładka 'Zbieranie danych':\n"
                             "   - Wybierz kamerę, ustaw literę/cyfrę i zbieraj dane (punkty dłoni).\n"
                             "   - Spacja/Enter lub przycisk 'Zapisz dane' zapisuje dane do CSV oraz obraz do folderu images/.\n"
                             "   - Ustawienia obrazu i MediaPipe wpływają na jakość detekcji.\n"
                             "   - 'Wyczyść folder images' usuwa wszystkie zebrane obrazy, 'Wyzeruj plik CSV' resetuje dane.\n"
                             "   - 'Przywróć domyślne' – resetuje wszystkie ustawienia.\n\n"
                             "2) Zakładka 'Trening modelu':\n"
                             "   - Ustaw plik CSV, model, scaler oraz parametry treningu.\n"
                             "   - Po kliknięciu 'Rozpocznij trening' trening uruchamia się w tle, a postęp epok jest aktualizowany.\n\n"
                             "3) Zakładka 'Detekcja znaków':\n"
                             "   - Po kliknięciu 'Start Detekcji' uruchamia się moduł rozpoznawania znaków z kamery.\n"
                             "   - Obraz z kamery jest wyświetlany, a rozpoznane znaki pojawiają się w polu tekstowym.\n"
                             "   - Przycisk 'Stop Detekcji' zatrzymuje detekcję.\n\n"
                             "4) Zakładka 'Instrukcja':\n"
                             "   - Znajdziesz tutaj pełen opis działania aplikacji.\n\n"
                             "Na dole aplikacji znajduje się wspólne pole logów.\n"
                             "Miłego korzystania!\n")
        label_instruct = ttk.Label(self.tab_instructions, text=instructions_text, justify=tk.LEFT)
        label_instruct.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=10)

    def start_detection(self):
        if self.detection_running:
            return
        self.detection_running = True
        self.start_det_button.config(state="disabled")
        self.stop_det_button.config(state="normal")
        self.det_thread = threading.Thread(target=self.run_detection)
        self.det_thread.start()

    def stop_detection(self):
        self.detection_running = False
        self.start_det_button.config(state="normal")
        self.stop_det_button.config(state="disabled")

    def run_detection(self):
        model = tf.keras.models.load_model(self.model_file_var.get())
        scaler = joblib.load(self.scaler_file_var.get())
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        mp_drawing = mp.solutions.drawing_utils
        classes = ['A', 'B', 'C']
        cap = cv2.VideoCapture(self.current_camera_index)
        while self.detection_running:
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            letter = ""
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
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            frame_rgb_for_tk = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb_for_tk)
            imgtk = ImageTk.PhotoImage(image=img)
            self.det_camera_label.imgtk = imgtk
            self.det_camera_label.configure(image=imgtk)
            if letter:
                self.det_text.insert(tk.END, letter + "\n")
                self.det_text.see(tk.END)
        cap.release()
        hands.close()

    def update_frame(self):
        current_tab_index = self.notebook.index(self.notebook.select())
        if current_tab_index == 0 and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.flip_vertical:
                    frame = cv2.flip(frame, 1)
                frame = self.apply_image_adjustments(frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                if self.current_label is not None:
                    cv2.putText(frame, f'Litera/Liczba: {self.current_label}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                index_display = str(self.new_index) if self.new_index is not None else 'N/A'
                cv2.putText(frame, f'Indeks znaku (ostatniego): {index_display}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                self.last_frame = frame.copy()
                frame_rgb_for_tk = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb_for_tk)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
        self.root.after(20, self.update_frame)

    def apply_image_adjustments(self, frame):
        frame = frame.astype(np.float32)
        beta = float(self.brightness_var.get())
        alpha_percent = float(self.contrast_var.get())
        alpha = max(0.01, alpha_percent / 100.0)
        frame = frame * alpha + beta
        shift_r = float(self.r_var.get())
        shift_g = float(self.g_var.get())
        shift_b = float(self.b_var.get())
        frame[:, :, 2] += shift_r
        frame[:, :, 1] += shift_g
        frame[:, :, 0] += shift_b
        frame = np.clip(frame, 0, 255)
        frame = frame / 255.0
        real_gamma = max(0.01, float(self.gamma_var.get()) / 100.0)
        frame = np.power(frame, 1.0 / real_gamma)
        frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def quit_app(self):
        self.log("Zamykanie aplikacji...")
        if self.cap:
            self.cap.release()
        if self.log_file:
            self.log_file.close()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = HandDataCollectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
