import tkinter as tk
from tkinter import ttk

def create_train_tab(app):
    app.train_main_frame = ttk.Frame(app.tab_train)
    app.train_main_frame.pack(fill=tk.BOTH, expand=True)
    config_frame = ttk.LabelFrame(app.train_main_frame, text="Konfiguracja treningu")
    config_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
    ttk.Label(config_frame, text="Test size (np. 0.2):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    app.test_size_var = tk.DoubleVar(value=0.2)
    ttk.Entry(config_frame, textvariable=app.test_size_var, width=8).grid(row=0, column=1, sticky=tk.W)
    ttk.Label(config_frame, text="Random state:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    app.random_state_var = tk.IntVar(value=42)
    ttk.Entry(config_frame, textvariable=app.random_state_var, width=8).grid(row=1, column=1, sticky=tk.W)
    ttk.Label(config_frame, text="Epochs:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    app.epochs_var = tk.IntVar(value=50)
    ttk.Entry(config_frame, textvariable=app.epochs_var, width=8).grid(row=2, column=1, sticky=tk.W)
    ttk.Label(config_frame, text="Batch size:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
    app.batch_size_var = tk.IntVar(value=32)
    ttk.Entry(config_frame, textvariable=app.batch_size_var, width=8).grid(row=3, column=1, sticky=tk.W)
    ttk.Label(config_frame, text="Patience (EarlyStopping):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
    app.patience_var = tk.IntVar(value=5)
    ttk.Entry(config_frame, textvariable=app.patience_var, width=8).grid(row=4, column=1, sticky=tk.W)
    train_button_frame = ttk.Frame(app.train_main_frame)
    train_button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
    start_train_button = ttk.Button(train_button_frame, text="Rozpocznij trening", command=app.start_training)
    start_train_button.pack(side=tk.LEFT, padx=5)
    from utils import disable_space_activation
    disable_space_activation(start_train_button)
    app.progress_var = tk.DoubleVar(value=0.0)
    app.progress_bar = ttk.Progressbar(app.train_main_frame, variable=app.progress_var,
                                       orient="horizontal", length=300, mode="determinate")
    app.progress_bar.pack(side=tk.TOP, padx=10, pady=5)
