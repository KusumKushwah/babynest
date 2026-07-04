import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from pushbullet import Pushbullet

# TensorFlow environment compilation validation
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


class BabyNestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BabyNest ML Console")
        self.root.geometry("600x500")
        self.root.configure(bg="#f8f9fa")

        # Pushbullet token is read from an environment variable.
        # Set it before running: export PUSHBULLET_TOKEN="your_token_here"
        self.pushbullet_token = os.environ.get("PUSHBULLET_TOKEN")
        if not self.pushbullet_token:
            print("WARNING: PUSHBULLET_TOKEN not set. Notifications will be disabled.")

        self.model_path = "babynest_model.keras"

        self.class_names = ['sleeping', 'crying', 'awake', 'empty_cradle']

        self.load_model()
        self.setup_ui()

    def load_model(self):
        """Load the trained BabyNest model."""
        self.model = None
        if HAS_TF and self.model_path and os.path.exists(self.model_path):
            try:
                print(f"Loading model from '{self.model_path}'...")
                self.model = tf.keras.models.load_model(
                    self.model_path,
                    custom_objects={"AdamW": tf.keras.optimizers.AdamW}
                )
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Model load failed, falling back to demo mode: {e}")
        else:
            if not HAS_TF:
                print("WARNING: TensorFlow not installed.")
            if not os.path.exists(self.model_path):
                print(f"WARNING: Model file not found at '{self.model_path}'.")

    def setup_ui(self):
        tk.Label(self.root, text="BabyNest ML Engine",
                 font=("Arial", 16, "bold"), fg="#2c3e50", bg="#f8f9fa").pack(pady=10)

        initial_status = f"System Ready. Model: {self.model_path if self.model else 'NOT LOADED - demo mode'}"
        self.status_label = tk.Label(self.root, text=initial_status,
                                      font=("Arial", 11),
                                      fg="#27ae60" if self.model else "#c0392b", bg="#f8f9fa")
        self.status_label.pack(pady=10)

        self.capture_btn = tk.Button(self.root, text="Capture & Predict",
                                      command=self.process_webcam_and_predict,
                                      font=("Arial", 12, "bold"), bg="#27ae60", fg="white",
                                      padx=10, pady=5, activebackground="#2196F3")
        self.capture_btn.pack(pady=20)

        self.result_frame = tk.LabelFrame(self.root, text=" Live AI Inference Result ",
                                           font=("Arial", 10, "bold"), bg="#f8f9fa", padx=10, pady=10)
        self.result_frame.pack(pady=20, fill="x", padx=30)

        self.result_label = tk.Label(self.result_frame, text="Prediction: Waiting for Input...",
                                      font=("Arial", 14), bg="#f8f9fa", fg="#2c3e50")
        self.result_label.pack()

    def send_push_alert(self, prediction_class):
        if not self.pushbullet_token:
            print("Skipping alert: no Pushbullet token configured.")
            return
        try:
            pb = Pushbullet(self.pushbullet_token)
            title = "BabyNest Alert"
            body = f"Detected state: '{prediction_class.upper()}'. Please check on your baby."
            pb.push_note(title, body)
            self.status_label.config(text="Alert sent to linked device!", fg="#e74c3c")
        except Exception as e:
            print(f"Pushbullet alert failed: {e}")

    def process_webcam_and_predict(self):
        self.status_label.config(text="Opening webcam...", fg="#f39c12")
        self.root.update()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Webcam Error", "Could not access the webcam.")
            return

        ret = False
        frame = None
        for _ in range(15):
            ret, frame = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror("Capture Error", "Failed to read a frame from the webcam.")
            return

        self.status_label.config(text="Analyzing frame...", fg="#2980b9")
        self.root.update()

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(rgb_frame, (224, 224))
            img_array = np.array(resized_img, dtype=np.float32) / 255.0

            if self.model is not None:
                img_batch = np.expand_dims(img_array, axis=0)
                predictions = self.model.predict(img_batch)
                class_index = np.argmax(predictions[0])
                detected_status = self.class_names[class_index]
                confidence_score = predictions[0][class_index] * 100
                label_prefix = "Prediction"
            else:
                # DEMO MODE ONLY: no trained model available, so this is NOT a
                # real prediction. Clearly labeled to avoid being mistaken
                # for an actual model output during a demo.
                mean_pixel_weight = float(np.mean(img_array))
                index_router = int((mean_pixel_weight * 100) % len(self.class_names))
                detected_status = self.class_names[index_router]
                confidence_score = 50.0
                label_prefix = "[SIMULATED - NO MODEL LOADED]"

            self.result_label.config(
                text=f"{label_prefix}: {detected_status.upper()} ({confidence_score:.1f}%)",
                fg="#e74c3c" if detected_status == 'crying' else "#27ae60"
            )

            if self.model is not None and detected_status == 'crying':
                self.send_push_alert(detected_status)
            else:
                self.status_label.config(text=f"Status: {detected_status}.", fg="#27ae60")

        except Exception as e:
            messagebox.showerror("Inference Error", f"Prediction failed: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = BabyNestApp(root)
    root.mainloop()
