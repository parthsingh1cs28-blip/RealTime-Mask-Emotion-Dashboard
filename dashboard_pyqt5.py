import sys
import cv2
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
import winsound

# ── Paths ─────────────────────────────────────────────────────
MODEL_PATH = r"C:\Users\PARTH SINGH\RealTime-Mask-Emotion-Dashboard\training\models\mobilenetv2_mask_model.h5"

# ── Camera Thread (runs separately for zero lag) ──────────────
class CameraThread(QThread):
    frame_signal     = pyqtSignal(np.ndarray)
    prediction_signal = pyqtSignal(str, float)
    violation_signal  = pyqtSignal(str, str, str)

    def __init__(self):
        super().__init__()
        self.running      = False
        self.model        = load_model(MODEL_PATH)
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.last_beep    = 0
        self.frame_count  = 0
        self.last_faces   = []  # Remember last detected faces to avoid flicker

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        import time
        self.running = True

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect every 2nd frame, remember last faces to avoid flicker
            if self.frame_count % 2 == 0:
                detected = self.face_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80)
                )
                if len(detected) > 0:
                    self.last_faces = detected
                faces = self.last_faces
            else:
                faces = self.last_faces

            status     = "No Face"
            confidence = 0.0

            for (fx, fy, fw, fh) in faces:
                # More generous crop so VGG16 sees full face with mask
                pad_x = int(fw * 0.3)
                pad_y = int(fh * 0.3)
                x1 = max(0, fx - pad_x);      y1 = max(0, fy - pad_y)
                x2 = min(w, fx + fw + pad_x); y2 = min(h, fy + fh + pad_y)

                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue

                face_arr = np.expand_dims(cv2.resize(face_roi, (224, 224)) / 255.0, axis=0)
                pred     = self.model.predict(face_arr, verbose=0)[0][0]
                mask_on  = pred < 0.5
                conf     = float((1 - pred) if mask_on else pred)

                if mask_on:
                    label  = "With Mask"
                    color  = (46, 204, 113)
                    status = "With Mask"
                else:
                    label  = "No Mask"
                    color  = (231, 76, 60)
                    status = "No Mask"
                    confidence = conf

                    # Violation
                    now = datetime.now()
                    self.violation_signal.emit(
                        now.strftime("%H:%M:%S"),
                        now.strftime("%Y-%m-%d"),
                        f"{conf*100:.1f}%"
                    )

                    # Beep
                    if time.time() - self.last_beep > 2:
                        try: winsound.Beep(1000, 300)
                        except: pass
                        self.last_beep = time.time()

                confidence = conf

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                txt = f"{label} {conf*100:.1f}%"
                cv2.rectangle(frame, (x1, y1-30), (x1+len(txt)*11, y1), color, -1)
                cv2.putText(frame, txt, (x1+4, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            self.frame_signal.emit(frame)
            self.prediction_signal.emit(status, confidence)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


# ── Main Dashboard Window ─────────────────────────────────────
class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Face Mask Detection Dashboard — VGG16")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet("background-color: #0e1117; color: white;")

        self.camera_thread  = None
        self.mask_count     = 0
        self.nomask_count   = 0
        self.violation_rows = 0

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(15, 15, 15, 15)

        # ── Title ──────────────────────────────────────────────
        title = QLabel("😷  Real-Time Face Mask Detection Dashboard")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("""
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #1a1a2e, stop:0.5 #16213e, stop:1 #0f3460);
            border-radius: 10px;
            padding: 12px;
            color: white;
        """)
        root.addWidget(title)

        # ── Body ───────────────────────────────────────────────
        body = QHBoxLayout()
        root.addLayout(body)

        # Left — camera feed
        left = QVBoxLayout()
        body.addLayout(left, 3)

        cam_label_title = QLabel("📹  Live Camera Feed")
        cam_label_title.setFont(QFont("Arial", 11, QFont.Bold))
        cam_label_title.setStyleSheet("color:#aaa; padding:4px;")
        left.addWidget(cam_label_title)

        self.cam_label = QLabel()
        self.cam_label.setMinimumSize(640, 440)
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setStyleSheet("""
            background: #1e2130;
            border-radius: 10px;
            border: 2px dashed #3a3d4e;
        """)
        self.cam_label.setText("📷\n\nClick  ▶ Start Camera  to begin")
        self.cam_label.setFont(QFont("Arial", 13))
        left.addWidget(self.cam_label)

        # Right — stats
        right = QVBoxLayout()
        right.setSpacing(8)
        body.addLayout(right, 1)

        right.addWidget(self._section_label("📊  Live Stats"))

        self.status_box = self._metric_box("STATUS", "Idle", "#f39c12")
        right.addWidget(self.status_box[0])

        self.conf_box = self._metric_box("CONFIDENCE", "0.0%", "#3498db")
        right.addWidget(self.conf_box[0])

        self.mask_box   = self._metric_box("WITH MASK",    "0", "#2ecc71")
        self.nomask_box = self._metric_box("NO MASK",      "0", "#e74c3c")
        self.viol_box   = self._metric_box("VIOLATIONS",   "0", "#f39c12")
        right.addWidget(self.mask_box[0])
        right.addWidget(self.nomask_box[0])
        right.addWidget(self.viol_box[0])

        right.addWidget(self._section_label("ℹ️  Model Info"))
        info = QLabel("Model: VGG16\nTask: Mask Detection\nInput: 224×224\nClasses: Mask / No Mask")
        info.setStyleSheet("""
            background:#1e2130; border-radius:8px;
            padding:10px; color:#aaa; font-size:12px;
        """)
        right.addWidget(info)
        right.addStretch()

        # ── Alert Bar ──────────────────────────────────────────
        self.alert_label = QLabel("  Waiting for camera...")
        self.alert_label.setAlignment(Qt.AlignCenter)
        self.alert_label.setFont(QFont("Arial", 13, QFont.Bold))
        self.alert_label.setFixedHeight(48)
        self.alert_label.setStyleSheet("""
            background:#1e2130; border-radius:8px;
            color:#888; border:2px solid #3a3d4e;
        """)
        root.addWidget(self.alert_label)

        # ── Buttons ────────────────────────────────────────────
        btn_row = QHBoxLayout()
        root.addLayout(btn_row)

        self.start_btn = QPushButton("▶  Start Camera")
        self.stop_btn  = QPushButton("⏹  Stop Camera")
        self.clear_btn = QPushButton("🗑  Clear Log")

        for btn, color in [
            (self.start_btn, "#2ecc71"),
            (self.stop_btn,  "#e74c3c"),
            (self.clear_btn, "#f39c12"),
        ]:
            btn.setFixedHeight(40)
            btn.setFont(QFont("Arial", 11, QFont.Bold))
            btn.setStyleSheet(f"""
                QPushButton {{
                    background:{color}; color:white;
                    border-radius:8px; border:none;
                }}
                QPushButton:hover {{ background:{color}cc; }}
            """)
            btn_row.addWidget(btn)

        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.clear_btn.clicked.connect(self.clear_log)

        # ── Violation Log ──────────────────────────────────────
        root.addWidget(self._section_label("📋  Violation Log"))

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Time", "Date", "Confidence", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setFixedHeight(160)
        self.table.setStyleSheet("""
            QTableWidget {
                background:#1e2130; color:white;
                border-radius:8px; border:1px solid #3a3d4e;
                gridline-color:#2a2d3e;
            }
            QHeaderView::section {
                background:#2a2d3e; color:#aaa;
                padding:6px; border:none; font-weight:bold;
            }
            QTableWidget::item { padding:4px; }
        """)
        root.addWidget(self.table)

    # ── Helpers ───────────────────────────────────────────────
    def _section_label(self, text):
        lbl = QLabel(text)
        lbl.setFont(QFont("Arial", 10, QFont.Bold))
        lbl.setStyleSheet("color:#aaa; padding:2px 0;")
        return lbl

    def _metric_box(self, title, value, color):
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 #1e2130, stop:1 #2a2d3e);
                border-radius:10px;
                border:1px solid #3a3d4e;
            }}
        """)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        t = QLabel(title)
        t.setAlignment(Qt.AlignCenter)
        t.setStyleSheet("color:#888; font-size:11px; letter-spacing:1px; border:none; background:transparent;")

        v = QLabel(value)
        v.setAlignment(Qt.AlignCenter)
        v.setFont(QFont("Arial", 20, QFont.Bold))
        v.setStyleSheet(f"color:{color}; border:none; background:transparent;")

        layout.addWidget(t)
        layout.addWidget(v)
        return frame, v

    # ── Camera Controls ───────────────────────────────────────
    def start_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            return
        self.camera_thread = CameraThread()
        self.camera_thread.frame_signal.connect(self.update_frame)
        self.camera_thread.prediction_signal.connect(self.update_prediction)
        self.camera_thread.violation_signal.connect(self.log_violation)
        self.camera_thread.start()

    def stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
        self.cam_label.setText("📷\n\nCamera stopped.")
        self.alert_label.setText("  Camera stopped.")
        self.alert_label.setStyleSheet("""
            background:#1e2130; border-radius:8px;
            color:#888; border:2px solid #3a3d4e;
        """)

    def clear_log(self):
        self.table.setRowCount(0)
        self.violation_rows = 0
        self.mask_count     = 0
        self.nomask_count   = 0
        self.mask_box[1].setText("0")
        self.nomask_box[1].setText("0")
        self.viol_box[1].setText("0")

    # ── Slots ─────────────────────────────────────────────────
    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.cam_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.cam_label.width(), self.cam_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def update_prediction(self, status, confidence):
        self.conf_box[1].setText(f"{confidence*100:.1f}%")
        self.status_box[1].setText(status)

        if status == "With Mask":
            self.mask_count += 1
            self.mask_box[1].setText(str(self.mask_count))
            self.status_box[1].setStyleSheet("color:#2ecc71; border:none; background:transparent;")
            self.alert_label.setText("  ✅  Mask Detected — Thank You for Complying!")
            self.alert_label.setStyleSheet("""
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #2ecc7122, stop:1 #27ae6022);
                border-radius:8px; color:#2ecc71;
                border:2px solid #2ecc71; font-size:14px; font-weight:bold;
            """)
        elif status == "No Mask":
            self.nomask_count += 1
            self.nomask_box[1].setText(str(self.nomask_count))
            self.status_box[1].setStyleSheet("color:#e74c3c; border:none; background:transparent;")
            self.alert_label.setText("  🚨  ALERT — Face Mask Not Detected! Please Wear Your Mask!")
            self.alert_label.setStyleSheet("""
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #e74c3c22, stop:1 #c0392b22);
                border-radius:8px; color:#e74c3c;
                border:2px solid #e74c3c; font-size:14px; font-weight:bold;
            """)
        else:
            self.status_box[1].setStyleSheet("color:#f39c12; border:none; background:transparent;")
            self.alert_label.setText("  👤  Waiting for face detection...")
            self.alert_label.setStyleSheet("""
                background:#1e2130; border-radius:8px;
                color:#888; border:2px solid #3a3d4e;
            """)

    def log_violation(self, time_str, date_str, conf_str):
        self.violation_rows += 1
        self.viol_box[1].setText(str(self.violation_rows))

        row = 0
        self.table.insertRow(row)
        for col, val in enumerate([time_str, date_str, conf_str, "🔴 No Mask"]):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            item.setForeground(QColor("#e74c3c"))
            self.table.setItem(row, col, item)

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = Dashboard()
    win.show()
    sys.exit(app.exec_())
