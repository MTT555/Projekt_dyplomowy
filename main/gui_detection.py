import tkinter as tk
from tkinter import ttk, scrolledtext

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

    if not hasattr(app, 'enter_mode_var'):
        app.enter_mode_var = tk.BooleanVar(value=False)

    enter_checkbutton = ttk.Checkbutton(
        control_frame,
        text="Wstawiaj znak tylko po Enterze",
        variable=app.enter_mode_var
    )
    enter_checkbutton.pack(side=tk.TOP, anchor='w', padx=10, pady=5)

    btn_frame = ttk.Frame(control_frame)
    btn_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

    app.start_det_button = ttk.Button(btn_frame, text="Start Detekcji", command=app.start_detection)
    app.start_det_button.pack(side=tk.LEFT, padx=5)

    app.stop_det_button = ttk.Button(btn_frame, text="Stop Detekcji", command=app.stop_detection, state="disabled")
    app.stop_det_button.pack(side=tk.LEFT, padx=5)

    app.clear_screen_button = ttk.Button(
        btn_frame,
        text="Wyczyść ekran",
        command=lambda: app.det_text.delete('1.0', tk.END)
    )
    app.clear_screen_button.pack(side=tk.LEFT, padx=5)

    app.root.bind("<Return>", lambda event: on_enter(event, app))
