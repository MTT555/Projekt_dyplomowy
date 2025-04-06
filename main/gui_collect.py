import tkinter as tk
from tkinter import ttk
from utils import disable_space_activation

def create_collect_tab(app):
    app.collect_left_frame = ttk.Frame(app.tab_collect)
    app.collect_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    app.collect_right_frame = ttk.Frame(app.tab_collect, width=550)
    app.collect_right_frame.pack(side=tk.RIGHT, fill=tk.Y)
    app.camera_label = ttk.Label(app.collect_left_frame)
    app.camera_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    app.controls_frame = ttk.Frame(app.collect_right_frame)
    app.controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
    app.mp_frame = ttk.Frame(app.collect_right_frame)
    app.mp_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
    create_collect_controls(app)
    create_mediapipe_frame(app)

def create_collect_controls(app):
    ttk.Label(app.controls_frame, text="Wybierz kamerę:").pack(pady=(5,0))
    app.camera_combo = ttk.Combobox(app.controls_frame, state="readonly", values=app.available_cameras)
    app.camera_combo.set(app.current_camera_index)
    app.camera_combo.pack(padx=5, pady=5)
    app.camera_combo.bind("<<ComboboxSelected>>", app.on_camera_select)
    ttk.Label(app.controls_frame, text="Podaj literę/cyfrę do zbierania:").pack(pady=5)
    app.label_entry = ttk.Entry(app.controls_frame)
    app.label_entry.pack(padx=5, pady=5)
    set_label_button = ttk.Button(app.controls_frame, text="Ustaw literę", command=app.set_label)
    set_label_button.pack(pady=5)
    disable_space_activation(set_label_button)
    save_button = ttk.Button(app.controls_frame, text="Zapisz dane", command=app.save_data)
    save_button.pack(pady=5)
    disable_space_activation(save_button)
    flip_button = ttk.Button(app.controls_frame, text="Flip w pionie (tab)", command=app.toggle_flip)
    flip_button.pack(pady=5)
    disable_space_activation(flip_button)
    ttk.Label(app.controls_frame, text="--- Zarządzanie danymi ---").pack(pady=(10,0))
    clear_images_button = ttk.Button(app.controls_frame, text="Wyczyść folder images", command=app.clear_images)
    clear_images_button.pack(pady=5)
    disable_space_activation(clear_images_button)
    clear_csv_button = ttk.Button(app.controls_frame, text="Wyzeruj plik CSV", command=app.clear_csv)
    clear_csv_button.pack(pady=5)
    disable_space_activation(clear_csv_button)
    ttk.Label(app.controls_frame, text="--- Reset ustawień ---").pack(pady=(10,0))
    reset_button = ttk.Button(app.controls_frame, text="Przywróć domyślne", command=app.reset_to_defaults)
    reset_button.pack(pady=5)
    disable_space_activation(reset_button)
    quit_button = ttk.Button(app.controls_frame, text="Wyjdź (q)", command=app.quit_app)
    quit_button.pack(pady=5)
    disable_space_activation(quit_button)
    ttk.Label(app.controls_frame, text="--- Ustawienia obrazu ---").pack(pady=(10,5))
    ttk.Label(app.controls_frame, text="Jasność (beta):").pack()
    app.brightness_var = tk.IntVar(value=0)
    app.brightness_value_label = ttk.Label(app.controls_frame, text="0")
    app.brightness_value_label.pack()
    app.brightness_scale = ttk.Scale(app.controls_frame, from_=-100, to=100,
                                     variable=app.brightness_var, orient=tk.HORIZONTAL,
                                     command=lambda e: app.update_scale_label(app.brightness_var, app.brightness_value_label))
    app.brightness_scale.pack(padx=5)
    ttk.Label(app.controls_frame, text="Kontrast (alpha%):").pack()
    app.contrast_var = tk.IntVar(value=100)
    app.contrast_value_label = ttk.Label(app.controls_frame, text="100")
    app.contrast_value_label.pack()
    app.contrast_scale = ttk.Scale(app.controls_frame, from_=0, to=200,
                                   variable=app.contrast_var, orient=tk.HORIZONTAL,
                                   command=lambda e: app.update_scale_label(app.contrast_var, app.contrast_value_label))
    app.contrast_scale.pack(padx=5)
    ttk.Label(app.controls_frame, text="Gamma (1.0 = brak):").pack()
    app.gamma_var = tk.IntVar(value=100)
    app.gamma_value_label = ttk.Label(app.controls_frame, text="100")
    app.gamma_value_label.pack()
    app.gamma_scale = ttk.Scale(app.controls_frame, from_=1, to=300,
                                variable=app.gamma_var, orient=tk.HORIZONTAL,
                                command=lambda e: app.update_scale_label(app.gamma_var, app.gamma_value_label))
    app.gamma_scale.pack(padx=5)
    ttk.Label(app.controls_frame, text="Przesunięcie koloru (R, G, B):").pack()
    frame_rgb = ttk.Frame(app.controls_frame)
    frame_rgb.pack(padx=5, pady=5, fill=tk.X)
    ttk.Label(frame_rgb, text="R:").grid(row=0, column=0, sticky=tk.W)
    app.r_var = tk.IntVar(value=0)
    app.r_value_label = ttk.Label(frame_rgb, text="0")
    app.r_value_label.grid(row=0, column=2, sticky=tk.W, padx=(5,0))
    r_scale = ttk.Scale(frame_rgb, from_=-50, to=50,
                        variable=app.r_var, orient=tk.HORIZONTAL, length=120,
                        command=lambda e: app.update_scale_label(app.r_var, app.r_value_label))
    r_scale.grid(row=0, column=1, padx=5)
    ttk.Label(frame_rgb, text="G:").grid(row=1, column=0, sticky=tk.W)
    app.g_var = tk.IntVar(value=0)
    app.g_value_label = ttk.Label(frame_rgb, text="0")
    app.g_value_label.grid(row=1, column=2, sticky=tk.W, padx=(5,0))
    g_scale = ttk.Scale(frame_rgb, from_=-50, to=50,
                        variable=app.g_var, orient=tk.HORIZONTAL, length=120,
                        command=lambda e: app.update_scale_label(app.g_var, app.g_value_label))
    g_scale.grid(row=1, column=1, padx=5)
    ttk.Label(frame_rgb, text="B:").grid(row=2, column=0, sticky=tk.W)
    app.b_var = tk.IntVar(value=0)
    app.b_value_label = ttk.Label(frame_rgb, text="0")
    app.b_value_label.grid(row=2, column=2, sticky=tk.W, padx=(5,0))
    b_scale = ttk.Scale(frame_rgb, from_=-50, to=50,
                        variable=app.b_var, orient=tk.HORIZONTAL, length=120,
                        command=lambda e: app.update_scale_label(app.b_var, app.b_value_label))
    b_scale.grid(row=2, column=1, padx=5)

def create_mediapipe_frame(app):
    ttk.Label(app.mp_frame, text="--- Ustawienia MediaPipe ---").pack(pady=(10,5))
    app.static_check = ttk.Checkbutton(app.mp_frame,
                                       text="static_image_mode (True = zdjęcia statyczne)",
                                       variable=app.static_image_mode_var)
    app.static_check.pack(padx=5, pady=2)
    ttk.Label(app.mp_frame, text="max_num_hands:").pack()
    app.max_hands_label = ttk.Label(app.mp_frame, text=str(app.max_num_hands_var.get()))
    app.max_hands_label.pack()
    app.max_num_hands_scale = ttk.Scale(app.mp_frame, from_=1, to=4,
                                        variable=app.max_num_hands_var, orient=tk.HORIZONTAL,
                                        command=lambda e: app.update_scale_label(app.max_num_hands_var, app.max_hands_label))
    app.max_num_hands_scale.pack(padx=5, pady=(0,5))
    ttk.Label(app.mp_frame, text="model_complexity (0-2):").pack()
    app.model_complexity_label = ttk.Label(app.mp_frame, text=str(app.model_complexity_var.get()))
    app.model_complexity_label.pack()
    app.model_complexity_scale = ttk.Scale(app.mp_frame, from_=0, to=2,
                                           variable=app.model_complexity_var, orient=tk.HORIZONTAL,
                                           command=lambda e: app.update_scale_label(app.model_complexity_var, app.model_complexity_label))
    app.model_complexity_scale.pack(padx=5, pady=(0,5))
    ttk.Label(app.mp_frame, text="min_detection_confidence (%):").pack()
    app.min_detection_label = ttk.Label(app.mp_frame, text=str(app.min_detection_confidence_var.get()))
    app.min_detection_label.pack()
    app.min_detection_scale = ttk.Scale(app.mp_frame, from_=0, to=100,
                                        variable=app.min_detection_confidence_var, orient=tk.HORIZONTAL,
                                        command=lambda e: app.update_scale_label(app.min_detection_confidence_var, app.min_detection_label))
    app.min_detection_scale.pack(padx=5, pady=(0,5))
    ttk.Label(app.mp_frame, text="min_tracking_confidence (%):").pack()
    app.min_tracking_label = ttk.Label(app.mp_frame, text=str(app.min_tracking_confidence_var.get()))
    app.min_tracking_label.pack()
    app.min_tracking_scale = ttk.Scale(app.mp_frame, from_=0, to=100,
                                       variable=app.min_tracking_confidence_var, orient=tk.HORIZONTAL,
                                       command=lambda e: app.update_scale_label(app.min_tracking_confidence_var, app.min_tracking_label))
    app.min_tracking_scale.pack(padx=5, pady=(0,5))
    apply_mediapipe_button = ttk.Button(app.mp_frame,
                                        text="Zastosuj zmiany w MediaPipe",
                                        command=app.update_mediapipe_settings)
    apply_mediapipe_button.pack(padx=5, pady=5)
    disable_space_activation(apply_mediapipe_button)
    file_label_frame = ttk.LabelFrame(app.mp_frame, text="Nazwa plików (domyślne wartości)")
    file_label_frame.pack(fill=tk.X, pady=(10,0))
    ttk.Label(file_label_frame, text="CSV (wej./wyj.):").grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
    csv_entry = ttk.Entry(file_label_frame, textvariable=app.csv_file_var, width=30)
    csv_entry.grid(row=0, column=1, sticky=tk.W, padx=2, pady=2)
    ttk.Label(file_label_frame, text="Model (wyj.):").grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
    model_entry = ttk.Entry(file_label_frame, textvariable=app.model_file_var, width=30)
    model_entry.grid(row=1, column=1, sticky=tk.W, padx=2, pady=2)
    ttk.Label(file_label_frame, text="Scaler (wyj.):").grid(row=2, column=0, sticky=tk.W, padx=2, pady=2)
    scaler_entry = ttk.Entry(file_label_frame, textvariable=app.scaler_file_var, width=30)
    scaler_entry.grid(row=2, column=1, sticky=tk.W, padx=2, pady=2)
