import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import threading

class AttendanceSystem:
    FACE_SIZE = (200, 200)
    LBPH_RADIUS = 2
    LBPH_NEIGHBORS = 8
    LBPH_GRID_X = 8
    LBPH_GRID_Y = 8
    MIN_FACE_SIZE = (80, 80)
    MIN_SHARPNESS = 45.0
    RECOGNITION_THRESHOLD = 55
    CAPTURE_DELAY_SECONDS = 3
    CAPTURE_SAMPLE_TARGET = 50
    CAPTURE_FRAME_INTERVAL = 2
    REQUIRED_STABLE_FRAMES = 5
    CAMERA_WARMUP_TIMEOUT = 5

    def __init__(self, root):
        self.root = root
        self.root.title("Attendance System (Haar + LBPH)")
        self.root.geometry("1180x720")
        self.root.minsize(980, 620)
        self.root.configure(background='#f4f6f8')
        self.worker_running = False
        self.camera_running = False
        self.camera = None
        self.camera_after_id = None
        self.camera_window = None

        # variables
        self.face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.recognizer = self.create_recognizer()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # check directories
        self.ensure_directory_structure()
        
        self.setup_gui()
        
        # load attendance if exist
        self.load_todays_attendance()

    def ensure_directory_structure(self):
        paths = ["TrainingImage", "TrainingImageLabel", "StudentDetails", "Attendance"]
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
        
        if not os.path.isfile("haarcascade_frontalface_default.xml"):
            messagebox.showerror("Error", "haarcascade_frontalface_default.xml is missing!")
            self.root.destroy()

    def create_recognizer(self):
        return cv2.face.LBPHFaceRecognizer_create(
            radius=self.LBPH_RADIUS,
            neighbors=self.LBPH_NEIGHBORS,
            grid_x=self.LBPH_GRID_X,
            grid_y=self.LBPH_GRID_Y
        )

    def preprocess_face(self, gray, x, y, w, h):
        """Normalize a detected face so capture, training, and prediction match."""
        pad_x = int(w * 0.08)
        pad_y = int(h * 0.12)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(gray.shape[1], x + w + pad_x)
        y2 = min(gray.shape[0], y + h + pad_y)

        face = gray[y1:y2, x1:x2]
        if face.size == 0:
            return None

        face = cv2.resize(face, self.FACE_SIZE)
        face = self.clahe.apply(face)
        return cv2.GaussianBlur(face, (3, 3), 0)

    def is_clear_face(self, face):
        if face is None:
            return False
        sharpness = cv2.Laplacian(face, cv2.CV_64F).var()
        return sharpness >= self.MIN_SHARPNESS

    def safe_filename_part(self, value):
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_")

    def open_camera(self, purpose):
        try:
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                cam.release()
                self.set_status("Could not open camera. Check camera permission and close other camera apps.")
                return None

            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.set_status(f"Camera opened for {purpose}.")
            return cam
        except Exception as e:
            self.set_status(f"Camera error: {e}")
            return None

    def setup_gui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f4f6f8")
        style.configure("Card.TFrame", background="#ffffff", relief="flat")
        style.configure("Title.TLabel", background="#f4f6f8", foreground="#17202a", font=("Arial", 24, "bold"))
        style.configure("Subtitle.TLabel", background="#f4f6f8", foreground="#5d6d7e", font=("Arial", 11))
        style.configure("CardTitle.TLabel", background="#ffffff", foreground="#17202a", font=("Arial", 15, "bold"))
        style.configure("TLabel", background="#ffffff", foreground="#17202a", font=("Arial", 11))
        style.configure("Status.TLabel", background="#ffffff", foreground="#1f618d", font=("Arial", 11, "bold"))
        style.configure("Primary.TButton", font=("Arial", 11, "bold"), padding=(14, 8))
        style.configure("Accent.TButton", font=("Arial", 12, "bold"), padding=(14, 10))
        style.configure("Treeview", font=("Arial", 10), rowheight=28)
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"))

        self.status_var = tk.StringVar(value="Ready")

        container = ttk.Frame(self.root, padding=24)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=2, uniform="main")
        container.columnconfigure(1, weight=3, uniform="main")
        container.rowconfigure(1, weight=1)

        header = ttk.Frame(container)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 18))
        ttk.Label(header, text="Face Recognition Attendance", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Register students, train the recognizer, then track attendance from the camera.",
                  style="Subtitle.TLabel").pack(anchor="w", pady=(4, 0))

        register_card = ttk.Frame(container, style="Card.TFrame", padding=22)
        register_card.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        register_card.columnconfigure(1, weight=1)

        ttk.Label(register_card, text="Register Student", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 20))
        ttk.Label(register_card, text="Student ID").grid(row=1, column=0, sticky="w", pady=8)
        self.txt_id = ttk.Entry(register_card, font=("Arial", 12))
        self.txt_id.grid(row=1, column=1, sticky="ew", pady=8, padx=(14, 0))

        ttk.Label(register_card, text="Name").grid(row=2, column=0, sticky="w", pady=8)
        self.txt_name = ttk.Entry(register_card, font=("Arial", 12))
        self.txt_name.grid(row=2, column=1, sticky="ew", pady=8, padx=(14, 0))

        actions = ttk.Frame(register_card, style="Card.TFrame")
        actions.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(18, 12))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

        self.take_btn = ttk.Button(actions, text="Take Images", command=self.take_images, style="Primary.TButton")
        self.take_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.train_btn = ttk.Button(actions, text="Train Model", command=self.train_images, style="Primary.TButton")
        self.train_btn.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self.clear_train_btn = ttk.Button(register_card, text="Clear Training Data", command=self.confirm_clear_training, style="Primary.TButton")
        self.clear_train_btn.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(18, 0))

        ttk.Label(register_card, textvariable=self.status_var, style="Status.TLabel", wraplength=360).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )

        attendance_card = ttk.Frame(container, style="Card.TFrame", padding=22)
        attendance_card.grid(row=1, column=1, sticky="nsew", padx=(12, 0))
        attendance_card.rowconfigure(2, weight=1)
        attendance_card.columnconfigure(0, weight=1)

        ttk.Label(attendance_card, text="Attendance Log", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.track_btn = ttk.Button(attendance_card, text="Track & Mark Attendance", command=self.track_images, style="Accent.TButton")
        self.track_btn.grid(row=1, column=0, sticky="ew", pady=(18, 14))

        self.clear_attendance_btn = ttk.Button(attendance_card, text="Clear Today's Attendance", command=self.clear_today_attendance, style="Primary.TButton")
        self.clear_attendance_btn.grid(row=2, column=0, sticky="ew", pady=(0, 14))

        self.delete_attendance_btn = ttk.Button(attendance_card, text="Delete Selected Entry", command=self.delete_selected_attendance, style="Primary.TButton")
        self.delete_attendance_btn.grid(row=3, column=0, sticky="ew", pady=(0, 14))

        table_frame = ttk.Frame(attendance_card, style="Card.TFrame")
        table_frame.grid(row=2, column=0, sticky="nsew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self.tv = ttk.Treeview(table_frame, columns=('ID', 'Name', 'Date', 'Time'), show='headings', height=15)
        self.tv.column('ID', width=80, anchor='center')
        self.tv.column('Name', width=180, anchor='w')
        self.tv.column('Date', width=130, anchor='center')
        self.tv.column('Time', width=110, anchor='center')
        self.tv.heading('ID', text='ID')
        self.tv.heading('Name', text='Name')
        self.tv.heading('Date', text='Date')
        self.tv.heading('Time', text='Time')
        self.tv.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tv.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tv.configure(yscrollcommand=scrollbar.set)

    def set_status(self, text):
        self.root.after(0, self.status_var.set, text)

    def set_busy(self, busy):
        def update_buttons():
            state = "disabled" if busy else "normal"
            self.take_btn.configure(state=state)
            self.train_btn.configure(state=state)
            self.track_btn.configure(state=state)
        self.root.after(0, update_buttons)

    def run_worker(self, target, *args):
        if self.worker_running or self.camera_running:
            self.set_status("Another task is already running.")
            return

        self.worker_running = True
        self.set_busy(True)

        def worker():
            try:
                target(*args)
            except Exception as e:
                self.set_status(f"Error: {e}")
            finally:
                cv2.destroyAllWindows()
                self.worker_running = False
                self.set_busy(False)

        threading.Thread(target=worker, daemon=True).start()

    def begin_camera_task(self):
        if self.worker_running or self.camera_running:
            self.set_status("Another task is already running.")
            return False

        self.camera_running = True
        self.set_busy(True)
        return True

    def finish_camera_task(self, status=None):
        if self.camera_after_id is not None:
            try:
                self.root.after_cancel(self.camera_after_id)
            except tk.TclError:
                pass
            self.camera_after_id = None

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        if self.camera_window:
            try:
                cv2.destroyWindow(self.camera_window)
            except cv2.error:
                cv2.destroyAllWindows()
            self.camera_window = None
        else:
            cv2.destroyAllWindows()

        self.camera_running = False
        self.set_busy(False)
        if status:
            self.set_status(status)

    # Func 1: Load Table on Startup
    def load_todays_attendance(self):
        """Reads today's CSV and populates the Treeview."""
        for item in self.tv.get_children():
            self.tv.delete(item)

        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        file_path = os.path.join("Attendance", f"Attendance_{date}.csv")
        
        if os.path.isfile(file_path):
            try:
                # Read CSV file
                with open(file_path, 'r') as f:
                    reader = csv.reader(f)
                    next(reader, None) # skip header
                    for row in reader:
                        if len(row) == 4: # row is valid
                            self.tv.insert('', 0, values=row)
            except Exception as e:
                print(f"Error loading CSV: {e}")

    def delete_training_images(self):
        path = "TrainingImage"
        removed = 0
        for filename in os.listdir(path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(path, filename)
                try:
                    os.remove(filepath)
                    removed += 1
                except OSError:
                    pass
        return removed

    def confirm_clear_training(self):
        if messagebox.askyesno("Confirm", "Hapus semua file training di TrainingImage sekarang?"):
            removed = self.delete_training_images()
            self.set_status(f"Deleted {removed} training images.")

    def clear_today_attendance(self):
        date = datetime.datetime.now().strftime('%Y-%m-%d')
        file_path = os.path.join("Attendance", f"Attendance_{date}.csv")
        if not os.path.isfile(file_path):
            self.set_status("Tidak ada file absensi hari ini untuk dihapus.")
            return

        if not messagebox.askyesno("Confirm", "Hapus file absensi hari ini?"):
            return

        try:
            os.remove(file_path)
            self.load_todays_attendance()
            self.set_status("Absensi hari ini berhasil dihapus.")
        except Exception as e:
            self.set_status(f"Could not delete attendance: {e}")

    def delete_selected_attendance(self):
        selected = self.tv.selection()
        if not selected:
            self.set_status("Pilih baris absensi yang ingin dihapus.")
            return

        item = selected[0]
        values = self.tv.item(item, 'values')
        if len(values) != 4:
            self.set_status("Data absensi tidak valid.")
            return

        user_id, name, date, timestamp = values
        file_path = os.path.join("Attendance", f"Attendance_{date}.csv")
        if not os.path.isfile(file_path):
            self.set_status("File absensi untuk tanggal ini tidak ditemukan.")
            return

        if not messagebox.askyesno("Confirm", f"Hapus absensi {name} ({user_id}) pada {date} {timestamp}?"):
            return

        try:
            rows = []
            removed = False
            with open(file_path, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if row == [user_id, name, date, timestamp] and not removed:
                        removed = True
                        continue
                    rows.append(row)

            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                if header:
                    writer.writerow(header)
                writer.writerows(rows)

            if removed:
                self.tv.delete(item)
                self.set_status(f"Absensi {name} berhasil dihapus.")
            else:
                self.set_status("Baris absensi tidak ditemukan di file.")
        except Exception as e:
            self.set_status(f"Could not delete selected attendance: {e}")

    # func 2: take images
    def take_images(self):
        user_id = self.txt_id.get().strip()
        name = self.txt_name.get().strip()

        if not user_id or not name:
            self.set_status("Enter ID and Name.")
            return

        if not user_id.isdigit():
            self.set_status("ID must be numeric.")
            return
        
        # Check for duplicates
        csv_path = os.path.join("StudentDetails", "StudentDetails.csv")
        if os.path.isfile(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0] == user_id:
                        self.set_status(f"ID {user_id} already exists.")
                        return

        safe_name = self.safe_filename_part(name) or "student"
        if not self.begin_camera_task():
            return

        self.camera = self.open_camera("image capture")
        if self.camera is None:
            self.finish_camera_task()
            return

        self.capture_user_id = user_id
        self.capture_name = name
        self.capture_safe_name = safe_name
        self.capture_csv_path = csv_path
        self.capture_sample_num = 0
        self.capture_frame_num = 0
        self.capture_start = time.time() + self.CAPTURE_DELAY_SECONDS
        self.camera_window = "Capturing Faces - Press Q to Stop"
        self.set_status("Camera ready. Position your face for the countdown.")
        self.process_capture_frame()

    def process_capture_frame(self):
        if not self.camera_running or self.camera is None:
            return

        ret, img = self.camera.read()
        if not ret:
            self.finish_capture("Camera stopped returning frames during capture.")
            return

        self.capture_frame_num += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=self.MIN_FACE_SIZE
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = self.preprocess_face(gray, x, y, w, h)

            # Keep samples varied and clear instead of saving many near-duplicate frames.
            if (
                time.time() >= self.capture_start
                and self.capture_frame_num % self.CAPTURE_FRAME_INTERVAL == 0
                and self.is_clear_face(face)
            ):
                self.capture_sample_num += 1
                cv2.imwrite(
                    f"TrainingImage/{self.capture_safe_name}.{self.capture_user_id}.{self.capture_sample_num}.jpg",
                    face
                )

        remaining = int(self.capture_start - time.time()) + 1
        if remaining > 0:
            cv2.putText(img, f"Starting capture in {remaining}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            if self.capture_frame_num % 20 == 0:
                self.set_status(f"Capturing images... {self.capture_sample_num}/{self.CAPTURE_SAMPLE_TARGET}")
            cv2.putText(img, f"Capturing: {self.capture_sample_num}/{self.CAPTURE_SAMPLE_TARGET}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(self.camera_window, img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or self.capture_sample_num >= self.CAPTURE_SAMPLE_TARGET:
            self.finish_capture()
            return

        self.camera_after_id = self.root.after(30, self.process_capture_frame)

    def finish_capture(self, error_status=None):
        sample_num = self.capture_sample_num
        name = self.capture_name
        user_id = self.capture_user_id
        csv_path = self.capture_csv_path
        
        # Save details only if images were taken
        if error_status:
            status = error_status
        elif sample_num > 0:
            header = ['ID', 'Name']
            row = [user_id, name]
            
            new_file = not os.path.isfile(csv_path)
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if new_file: writer.writerow(header)
                writer.writerow(row)
            status = f"Saved {sample_num} images for {name} ({user_id})."
        else:
            status = "No clear face detected. Try better lighting and face the camera."

        self.finish_camera_task(status)

    # func 3: train model
    def train_images(self):
        self.run_worker(self.train_images_worker)

    def train_images_worker(self):
        self.set_status("Training model...")
        
        faces, ids = [], []
        path = "TrainingImage"
        
        try:
            image_paths = [os.path.join(path, f) for f in os.listdir(path)]
            for image_path in image_paths:
                if not image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                pil_image = Image.open(image_path).convert('L')
                image_np = np.array(pil_image, 'uint8')
                face_id = int(os.path.split(image_path)[-1].split(".")[1])
                faces.append(image_np)
                ids.append(face_id)
            
            if len(faces) == 0:
                self.set_status("No training images found.")
                return

            trainer_path = "TrainingImageLabel/Trainner.yml"
            temp_trainer_path = os.path.join("TrainingImageLabel", "Trainner.tmp.yml")

            self.recognizer = self.create_recognizer()
            if os.path.isfile(trainer_path):
                self.recognizer.read(trainer_path)
                self.recognizer.update(faces, np.array(ids))
                action = "Updated"
            else:
                self.recognizer.train(faces, np.array(ids))
                action = "Trained"

            self.recognizer.save(temp_trainer_path)

            # Validate before replacing the current model so tracking never loads a partial file.
            test_recognizer = self.create_recognizer()
            test_recognizer.read(temp_trainer_path)
            os.replace(temp_trainer_path, trainer_path)

            deleted = self.delete_training_images()
            if deleted > 0:
                self.set_status(f"{action} model with {len(faces)} images. Deleted {deleted} new training files.")
            else:
                self.set_status(f"{action} model with {len(faces)} images.")

            self.root.after(0, messagebox.showinfo, "Success", f"Model {action.lower()} successfully.")
            
        except Exception as e:
            if 'temp_trainer_path' in locals() and os.path.exists(temp_trainer_path):
                os.remove(temp_trainer_path)
            self.set_status(f"Training error: {e}")

    # func 4: track n log
    def track_images(self):
        trainer_path = "TrainingImageLabel/Trainner.yml"
        if not os.path.isfile(trainer_path):
            messagebox.showerror("Error", "Train the model first!")
            return
        if not os.path.isfile("StudentDetails/StudentDetails.csv"):
            messagebox.showerror("Error", "No student details found!")
            return

        if not self.begin_camera_task():
            return

        self.set_status("Opening camera for attendance tracking...")
        self.track_recognizer = self.create_recognizer()
        try:
            self.track_recognizer.read(trainer_path)
        except cv2.error as e:
            self.finish_camera_task()
            messagebox.showerror(
                "Invalid trained model",
                "The trained model file is corrupt or was created with old settings. Click Train Model to rebuild it."
            )
            self.set_status(f"Could not load trained model: {e}")
            return
        
        # Load student
        try:
            self.track_df = pd.read_csv("StudentDetails/StudentDetails.csv", dtype={'ID': str})
        except Exception as e:
            self.finish_camera_task()
            self.set_status(f"Could not load student details: {e}")
            return

        self.camera = self.open_camera("attendance tracking")
        if self.camera is None:
            self.finish_camera_task()
            return

        self.track_stable_id = None
        self.track_stable_count = 0
        self.track_frame_num = 0
        self.track_last_status = 0
        self.camera_window = "Taking Attendance - Press Q to Stop"
        self.set_status("Camera ready. Look at the camera and hold still.")
        self.process_tracking_frame()

    def process_tracking_frame(self):
        if not self.camera_running or self.camera is None:
            return

        ret, im = self.camera.read()
        if not ret:
            self.finish_camera_task("Camera stopped returning frames during tracking.")
            return

        self.track_frame_num += 1
                
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=self.MIN_FACE_SIZE
        )

        now = time.time()
        if now - self.track_last_status > 1.5:
            if len(faces) == 0:
                self.set_status("Tracking... no face detected yet. Press Q in the camera window to stop.")
            else:
                self.set_status("Tracking... hold still while the app verifies your face.")
            self.track_last_status = now
            
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            face = self.preprocess_face(gray, x, y, w, h)
            if face is None:
                continue

            serial_id, conf = self.track_recognizer.predict(face)
            
            if conf < self.RECOGNITION_THRESHOLD:
                name_series = self.track_df.loc[self.track_df['ID'] == str(serial_id)]['Name'].values
                name = name_series[0] if len(name_series) > 0 else "Unknown"
                
                if self.track_stable_id == serial_id:
                    self.track_stable_count += 1
                else:
                    self.track_stable_id = serial_id
                    self.track_stable_count = 1

                cv2.putText(im, f"{name} ({self.track_stable_count}/{self.REQUIRED_STABLE_FRAMES})",
                            (x, y + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                if self.track_stable_count >= self.REQUIRED_STABLE_FRAMES:
                    self.finish_camera_task(f"Recognized {name}. Attendance check completed.")
                    self.log_attendance(serial_id, name)
                    return
                
            else:
                self.track_stable_id = None
                self.track_stable_count = 0
                cv2.putText(im, "Unknown", (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow(self.camera_window, im)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            self.finish_camera_task("Tracking stopped.")
            return

        self.camera_after_id = self.root.after(30, self.process_tracking_frame)

    def log_attendance(self, user_id, name):
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        
        # Prevent Duplicate uis
        existing_items = self.tv.get_children()
        for item in existing_items:
            vals = self.tv.item(item)['values']
            if str(vals[0]) == str(user_id) and vals[2] == date:
                messagebox.showinfo("Info", "Attendance already logged for today!")
                return 

        # Add to CSV
        file_path = os.path.join("Attendance", f"Attendance_{date}.csv")
        exists = os.path.isfile(file_path)
        
        try:
            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not exists:
                    writer.writerow(['ID', 'Name', 'Date', 'Time'])
                writer.writerow([user_id, name, date, timestamp])
                
            # Update UI Table immediately
            self.tv.insert('', 0, values=(user_id, name, date, timestamp))
            messagebox.showinfo("Success", f"Attendance Marked for {name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not save attendance: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystem(root)
    root.mainloop()
