import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras
import threading
import time
import pygame

class EmotionDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Deteksi Emosi Wajah - CNN")
        self.root.geometry("1050x800")
        self.root.configure(bg='#2c3e50')

        self.label_map = {
            0: 'Angry',
            1: 'Fatigue',
            2: 'Fear',
            3: 'Pain',
            4: 'Sad',
            5: 'Surprise'
        }

        self.emotion_colors = {
            'Angry': '#e74c3c',
            'Fatigue': '#9b59b6',
            'Fear': '#e67e22',
            'Pain': '#c0392b',
            'Sad': '#3498db',
            'Surprise': '#f1c40f'
        }

        self.model = None
        self.cap = None
        self.is_running = False
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.alarm_file = None
        self.is_alarm_playing = False
        self.fatigue_detected = False
        self.fatigue_count = 0
        self.fatigue_threshold = 3

        pygame.mixer.init()
        self.setup_ui()

    def setup_ui(self):
        header_frame = tk.Frame(self.root, bg='#34495e', height=80)
        header_frame.pack(fill='x', padx=10, pady=(10, 0))
        header_frame.pack_propagate(False)

        title_label = tk.Label(header_frame, text="🎭 DETEKSI EMOSI WAJAH",
                               font=('Arial', 15, 'bold'),
                               bg='#34495e', fg='white')
        title_label.pack(expand=True)

        control_frame = tk.Frame(self.root, bg='#2c3e50')
        control_frame.pack(fill='x', padx=10, pady=10)

        combined_frame = tk.Frame(control_frame, bg='#2c3e50')
        combined_frame.pack(fill='x')

        tk.Button(combined_frame, text="Load Model (.h5/.keras)",
                  command=self.load_model,
                  bg='#3498db', fg='white',
                  font=('Arial', 11, 'bold'),
                  relief='flat', padx=20, pady=5).pack(side='left', padx=10, pady=10)

        self.model_status = tk.Label(combined_frame, text="Model: Belum dimuat",
                                     bg='#34495e', fg='#e74c3c',
                                     font=('Arial', 11))
        self.model_status.pack(side='left', padx=20)

        self.start_btn = tk.Button(combined_frame, text="▶ Start Kamera",
                                   command=self.start_camera,
                                   bg='#27ae60', fg='white',
                                   font=('Arial', 8, 'bold'),
                                   relief='flat', padx=15, pady=5)
        self.start_btn.pack(side='left', padx=5, pady=10)

        self.stop_btn = tk.Button(combined_frame, text="⏹ Stop Kamera",
                                  command=self.stop_camera,
                                  bg='#e74c3c', fg='white',
                                  font=('Arial', 8, 'bold'),
                                  relief='flat', padx=15, pady=5,
                                  state='disabled')
        self.stop_btn.pack(side='left', padx=5, pady=10)

        self.alarm_btn = tk.Button(combined_frame, text="🔊 Load Alarm",
                                   command=self.load_alarm,
                                   bg='#f39c12', fg='white',
                                   font=('Arial', 8, 'bold'),
                                   relief='flat', padx=15, pady=5)
        self.alarm_btn.pack(side='left', padx=5, pady=10)

        self.alarm_status = tk.Label(combined_frame, text="Alarm: Belum dimuat",
                                     bg='#34495e', fg='#e74c3c',
                                     font=('Arial', 8))
        self.alarm_status.pack(side='left', padx=10)

        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        video_frame = tk.LabelFrame(main_frame, text="Video Feed",
                                    font=('Arial', 12, 'bold'),
                                    bg='#34495e', fg='white',
                                    relief='raised', bd=2)
        video_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))

        self.video_label = tk.Label(video_frame, bg='black',
                                    text="Kamera tidak aktif\n\nSilakan load model terlebih dahulu\nkemudian klik Start Kamera",
                                    fg='white', font=('Arial', 14))
        self.video_label.pack(fill='both', expand=True, padx=10, pady=10)

        results_frame = tk.LabelFrame(main_frame, text="Hasil Deteksi",
                                      font=('Arial', 12, 'bold'),
                                      bg='#34495e', fg='white',
                                      relief='raised', bd=2,
                                      width=250, height=500)
        results_frame.pack(side='right', fill='y', padx=(5, 0))
        results_frame.pack_propagate(False)

        self.current_emotion_label = tk.Label(results_frame,
                                              text="Emosi: -",
                                              font=('Arial', 16, 'bold'),
                                              bg='#34495e', fg='white')
        self.current_emotion_label.pack(pady=20)

        self.confidence_label = tk.Label(results_frame,
                                         text="Confidence: -",
                                         font=('Arial', 12),
                                         bg='#34495e', fg='white')
        self.confidence_label.pack(pady=10)

        self.fatigue_alert = tk.Label(results_frame,
                                      text="",
                                      font=('Arial', 12, 'bold'),
                                      bg='#34495e', fg='#e74c3c')
        self.fatigue_alert.pack(pady=10)

        tk.Label(results_frame, text="Probabilitas:",
                 font=('Arial', 12, 'bold'),
                 bg='#34495e', fg='white').pack(pady=(20, 10))

        self.emotion_vars = {}
        self.emotion_bars = {}

        for emotion in self.label_map.values():
            frame = tk.Frame(results_frame, bg='#34495e')
            frame.pack(fill='x', padx=10, pady=2)

            tk.Label(frame, text=emotion, bg='#34495e', fg='white',
                     font=('Arial', 10), width=8, anchor='w').pack(side='left')

            self.emotion_vars[emotion] = tk.DoubleVar()
            bar = ttk.Progressbar(frame, variable=self.emotion_vars[emotion],
                                  maximum=100, length=120)
            bar.pack(side='right')
            self.emotion_bars[emotion] = bar

        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Load model untuk memulai")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                              relief='sunken', anchor='w',
                              bg='#34495e', fg='white')
        status_bar.pack(side='bottom', fill='x')

    def load_alarm(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Pilih File Alarm",
                filetypes=[("Audio files", "*.mp3 *.wav *.ogg"), ("All files", "*.*")]
            )
            if file_path:
                pygame.mixer.music.load(file_path)
                self.alarm_file = file_path
                self.alarm_status.config(text="Alarm: Ready ✓", fg='#27ae60')
                self.status_var.set(f"Alarm loaded: {file_path.split('/')[-1]}")
                messagebox.showinfo("Success", "File alarm berhasil dimuat!")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat file alarm:\n{str(e)}")
            self.alarm_status.config(text="Alarm: Error", fg='#e74c3c')

    def play_alarm(self):
        if self.alarm_file and not self.is_alarm_playing:
            try:
                pygame.mixer.music.load(self.alarm_file)
                pygame.mixer.music.play(-1)
                self.is_alarm_playing = True
                self.fatigue_alert.config(
                    text="⚠️ KELELAHAN TERDEKSI!\n🔊 ALARM AKTIF",
                    fg='#e74c3c'
                )
            except Exception as e:
                print(f"Error playing alarm: {e}")

    def stop_alarm(self):
        if self.is_alarm_playing:
            try:
                pygame.mixer.music.stop()
                self.is_alarm_playing = False
                self.fatigue_alert.config(text="", fg='#27ae60')
            except Exception as e:
                print(f"Error stopping alarm: {e}")

    def check_fatigue_alarm(self, emotion, confidence):
        """Check if fatigue is detected and handle alarm"""
        if emotion == 'Fatigue' and confidence > 50:  # Threshold confidence 50%
            self.fatigue_count += 1
            if self.fatigue_count >= self.fatigue_threshold:
                if not self.fatigue_detected:
                    self.fatigue_detected = True
                    if self.alarm_file:
                        self.play_alarm()
                        self.status_var.set("⚠️ KELELAHAN TERDETEKSI - ALARM AKTIF!")
        else:
            # Reset fatigue detection if other emotion detected
            if self.fatigue_detected and emotion != 'Fatigue':
                self.fatigue_count = 0
                self.fatigue_detected = False
                self.stop_alarm()
                self.status_var.set("Kondisi normal - Alarm dihentikan")

    def load_model(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Pilih Model File",
                filetypes=[("Model files", "*.h5 *.keras"), ("All files", "*.*")]
            )
            if file_path:
                self.model = keras.models.load_model(file_path)
                self.model_status.config(text=f"Model: Loaded ✓", fg='#27ae60')
                self.status_var.set(f"Model loaded: {file_path.split('/')[-1]}")
                messagebox.showinfo("Success", "Model berhasil dimuat!")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat model:\n{str(e)}")
            self.status_var.set("Error loading model")

    def preprocess_face(self, face_img):
        try:
            enhanced_face = cv2.equalizeHist(face_img)
            face_resized = cv2.resize(enhanced_face, (80, 80))
            face_normalized = face_resized.astype('float32') / 255.0
            face_batch = np.expand_dims(face_normalized, axis=0)
            face_batch = np.expand_dims(face_batch, axis=-1)
            return face_batch
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None

    def predict_emotion(self, face_img):
        if self.model is None:
            return None, None, None
        try:
            processed_face = self.preprocess_face(face_img)
            predictions = self.model.predict(processed_face, verbose=0)[0]
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class] * 100
            emotion = self.label_map[predicted_class]
            return emotion, confidence, predictions
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None, None, None

    def update_emotion_display(self, emotion, confidence, predictions=None):
        if emotion and confidence:
            self.current_emotion_label.config(
                text=f"Emosi: {emotion}",
                fg=self.emotion_colors.get(emotion, 'white')
            )
            self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
            self.check_fatigue_alarm(emotion, confidence)
            if predictions is not None:
                for i, (label, prob) in enumerate(zip(self.label_map.values(), predictions)):
                    self.emotion_vars[label].set(prob * 100)
        else:
            self.current_emotion_label.config(text="Emosi: -", fg='white')
            self.confidence_label.config(text="Confidence: -")
            self.fatigue_count = 0
            if self.fatigue_detected:
                self.fatigue_detected = False
                self.stop_alarm()
            for emotion in self.emotion_vars:
                self.emotion_vars[emotion].set(0)

    def start_camera(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Harap load model terlebih dahulu!")
            return
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Tidak dapat mengakses kamera!")
                return
            self.is_running = True
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.status_var.set("Kamera aktif - Deteksi emosi berjalan")
            self.video_thread = threading.Thread(target=self.video_loop)
            self.video_thread.daemon = True
            self.video_thread.start()
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memulai kamera:\n{str(e)}")

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        if self.is_alarm_playing:
            self.stop_alarm()
        self.fatigue_detected = False
        self.fatigue_count = 0
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_var.set("Kamera dihentikan")
        self.video_label.config(image='', text="Kamera tidak aktif")
        self.update_emotion_display(None, None)

    def video_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            emotion_detected = False
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_gray = gray[y:y+h, x:x+w]
                emotion, confidence, predictions = self.predict_emotion(face_gray)
                if emotion and confidence:
                    color = self.emotion_colors.get(emotion, '#ffffff')
                    color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    cv2.putText(frame, f"{emotion} ({confidence:.1f}%)", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    self.root.after(0, lambda e=emotion, c=confidence, p=predictions:
                                    self.update_emotion_display(e, c, p))
                    emotion_detected = True
            if not emotion_detected:
                self.root.after(0, lambda: self.update_emotion_display(None, None))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((480, 360), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.root.after(0, lambda p=photo: self.update_video_display(p))
            time.sleep(0.03)

    def update_video_display(self, photo):
        self.video_label.config(image=photo, text='')
        self.video_label.image = photo

    def on_closing(self):
        if self.is_alarm_playing:
            self.stop_alarm()
        self.stop_camera()
        pygame.mixer.quit()
        self.root.destroy()

def main():
    pygame.init()
    pygame.mixer.init()
    root = tk.Tk()
    app = EmotionDetectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
