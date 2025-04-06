import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
class EpochProgressCallback(keras.callbacks.Callback):
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
        msg = (
            f"Epoch {current_epoch}/{self.total_epochs} - "
            f"loss: {logs.get('loss', 0):.4f}, "
            f"accuracy: {logs.get('accuracy', 0):.4f}, "
            f"val_loss: {logs.get('val_loss', 0):.4f}, "
            f"val_accuracy: {logs.get('val_accuracy', 0):.4f}"
        )
        self.root.after(0, lambda: self.app_log_func(msg))

def run_training_in_thread(app):
    csv_path = app.csv_file_var.get()
    model_path = app.model_file_var.get()
    scaler_path = app.scaler_file_var.get()
    test_size = app.test_size_var.get()
    random_state = app.random_state_var.get()
    epochs = app.epochs_var.get()
    batch_size = app.batch_size_var.get()
    patience = app.patience_var.get()

    if not os.path.exists(csv_path):
        app.log(f"Plik CSV {csv_path} nie istnieje – brak danych do trenowania!")
        return

    df = pd.read_csv(csv_path)
    if 'index' in df.columns:
        df.drop(columns=['index'], inplace=True)
    if 'label' not in df.columns:
        app.log("Brak kolumny 'label' w CSV! Nie można trenować.")
        return

    X = df.drop('label', axis=1)
    y = df['label']
    X, y = shuffle(X, y, random_state=random_state)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError as e:
        app.log(f"Błąd przy dzieleniu zbioru: {e}")
        return

    y_train_encoded = pd.get_dummies(y_train)
    y_test_encoded = pd.get_dummies(y_test)

    y_train_encoded = pd.get_dummies(y_train)
    label_map = {i: label for i, label in enumerate(y_train_encoded.columns)}
    print("Mapa klas:", label_map)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if not os.path.exists(os.path.dirname(scaler_path)):
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    app.log(f"Zapisano scaler do {scaler_path}")

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
    app.log(f"Model summary:\n{model.summary()}")

    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    progress_callback = EpochProgressCallback(
        total_epochs=epochs,
        progress_var=app.progress_var,
        app_log_func=app.log,
        root=app.root
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
    app.log(f"Zapisano model w {model_path}")

    test_loss, test_acc = model.evaluate(X_test_scaled, y_test_encoded, verbose=0)
    app.log(f"Test accuracy: {test_acc:.4f}")

    y_pred_prob = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test_encoded.values, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    app.log(f"Confusion matrix:\n{cm}")
    app.log(classification_report(y_true, y_pred))
    app.log("Trening zakończony.")
    app.root.after(0, lambda: app.progress_var.set(100.0))
