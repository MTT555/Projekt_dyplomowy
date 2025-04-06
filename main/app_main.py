import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import csv
import cv2
import numpy as np
import threading
import mediapipe as mp
import joblib
import tensorflow as tf
from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

import training
import detection
import gui_collect
import gui_train
import gui_detection
import gui_instructions
import gui_text_detection
from utils import disable_space_activation, TextRedirector

class HandDataCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikacja - Zbieranie danych, Trening i Detekcja")
        self.interval_var = tk.StringVar(value="1000")
        self.enter_mode_var = tk.BooleanVar(value=False)
        
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
        self.tab_text_detection = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_text_detection, text="Miganie tekstu")
        self.tab_instructions = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_instructions, text="Instrukcja")
        self.log_console = scrolledtext.ScrolledText(self.main_frame, height=10, wrap=tk.WORD,font=("Roboto", 12))
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
        
        gui_collect.create_collect_tab(self)
        gui_train.create_train_tab(self)
        gui_detection.create_detection_tab(self)
        gui_text_detection.create_text_detection_tab(self)
        gui_instructions.create_instructions_tab(self)
        
        
        self.root.bind_all("<space>", self._on_space_or_enter)
        self.root.bind_all("<Return>", self._on_space_or_enter)
        self.update_frame()
        
        self.detection_running = False

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
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                        )
                if self.current_label is not None:
                    cv2.putText(frame, f'Litera/Liczba: {self.current_label}', (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                index_display = str(self.new_index) if self.new_index is not None else 'N/A'
                cv2.putText(frame, f'Indeks znaku (ostatniego): {index_display}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
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

    def update_scale_label(self, variable: tk.IntVar, label: ttk.Label):
        val = variable.get()
        label.config(text=str(val))

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
        t = threading.Thread(target=training.run_training_in_thread, args=(self,))
        t.start()

    def start_detection(self):
        if self.detection_running:
            return
        self.detection_running = True
        self.start_det_button.config(state="disabled")
        self.stop_det_button.config(state="normal")
        t = threading.Thread(target=detection.run_detection, args=(self,))
        t.start()

    def stop_detection(self):
        self.detection_running = False
        self.start_det_button.config(state="normal")
        self.stop_det_button.config(state="disabled")

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

def main():
    root = tk.Tk()
    app = HandDataCollectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
