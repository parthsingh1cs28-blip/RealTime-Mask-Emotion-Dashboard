import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os, random
from collections import deque

# ── All Models ───────────────────────────────────────────────
MODELS = {
    "CNN"               : (r"C:\Users\PARTH SINGH\RealTime-Mask-Emotion-Dashboard\training\models\mask_detector_model.h5",    64),
    "MobileNetV2"       : (r"C:\Users\PARTH SINGH\RealTime-Mask-Emotion-Dashboard\training\models\mobilenetv2_mask_model.h5", 224),
    "VGG16"             : (r"C:\Users\PARTH SINGH\RealTime-Mask-Emotion-Dashboard\training\models\vgg16_mask_model.h5",       224),
    "ResNet50-FineTuned": (r"C:\Users\PARTH SINGH\RealTime-Mask-Emotion-Dashboard\training\models\resnet50_mask_finetuned.h5",224),
}

WITH_MASK_PATH    = r"C:\Users\PARTH SINGH\RealTime-Mask-Emotion-Dashboard\training\dataset\with_mask"
WITHOUT_MASK_PATH = r"C:\Users\PARTH SINGH\RealTime-Mask-Emotion-Dashboard\training\dataset\without_mask"
SAVE_DIR          = r"C:\Users\PARTH SINGH\RealTime-Mask-Emotion-Dashboard\training\models"
SAMPLES_EACH      = 4

# ── Face Detector ─────────────────────────────────────────────
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Helper: predict single face crop ─────────────────────────
def predict(model, face_crop, img_size):
    resized  = cv2.resize(face_crop, (img_size, img_size))
    arr      = np.expand_dims(resized / 255.0, axis=0)
    pred     = model.predict(arr, verbose=0)[0][0]
    mask_on  = pred < 0.5
    conf     = float((1 - pred) if mask_on else pred)
    label    = "With Mask" if mask_on else "No Mask"
    return label, conf

# ══════════════════════════════════════════════════════════════
# PART 1 — SAMPLE IMAGE TEST
# ══════════════════════════════════════════════════════════════
def test_on_images(model_name, model, img_size):
    print(f"\n── Sample Image Test: {model_name} ──")

    def get_samples(folder, n):
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        return random.sample(files, min(n, len(files)))

    with_files    = get_samples(WITH_MASK_PATH,    SAMPLES_EACH)
    without_files = get_samples(WITHOUT_MASK_PATH, SAMPLES_EACH)

    fig, axes = plt.subplots(2, SAMPLES_EACH, figsize=(SAMPLES_EACH * 3, 7))
    fig.suptitle(f'{model_name} — Sample Predictions', fontsize=15, fontweight='bold')

    def plot_row(files, folder, row, true_label):
        for col, fname in enumerate(files):
            path     = os.path.join(folder, fname)
            img      = image.load_img(path, target_size=(img_size, img_size))
            arr      = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
            pred     = model.predict(arr, verbose=0)[0][0]
            mask_on  = pred < 0.5
            conf     = float((1 - pred) if mask_on else pred)
            label    = "With Mask" if mask_on else "No Mask"
            correct  = (label == true_label)
            color    = '#2ecc71' if correct else '#e74c3c'
            icon     = '✔' if correct else '✘'

            display_img = image.load_img(path, target_size=(224, 224))
            axes[row, col].imshow(display_img)
            axes[row, col].axis('off')
            axes[row, col].set_title(
                f"{icon} {label}\n{conf*100:.1f}%",
                fontsize=9, color=color, fontweight='bold'
            )
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
                spine.set_visible(True)

    plot_row(with_files,    WITH_MASK_PATH,    0, "With Mask")
    plot_row(without_files, WITHOUT_MASK_PATH, 1, "No Mask")

    axes[0, 0].set_ylabel("With Mask\n(True)",    fontsize=10, fontweight='bold', color='#2980b9')
    axes[1, 0].set_ylabel("Without Mask\n(True)", fontsize=10, fontweight='bold', color='#e67e22')

    plt.tight_layout()
    save_path = f"{SAVE_DIR}\\{model_name.lower().replace('-','_')}_sample_test.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()

# ══════════════════════════════════════════════════════════════
# PART 2 — LIVE WEBCAM TEST
# ══════════════════════════════════════════════════════════════
def test_on_webcam(model_name, model, img_size):
    print(f"\n── Webcam Test: {model_name} ──")
    print("Press Q to quit and move to next model\n")

    cap              = cv2.VideoCapture(0)
    last_faces       = []
    frame_count      = 0
    prediction_buffer = deque(maxlen=5)  # Smoothing buffer

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (640, 480))
        h, w  = frame.shape[:2]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect every 2nd frame
        if frame_count % 2 == 0:
            detected = face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80)
            )
            if len(detected) > 0:
                last_faces = detected
        faces = last_faces

        for (fx, fy, fw, fh) in faces:

            # ── CNN uses tight crop + RGB conversion ──────────
            if model_name == "CNN":
                face_crop = frame[fy+10:fy+fh-10, fx+10:fx+fw-10]
                if face_crop.size == 0:
                    continue
                face_resized = cv2.resize(face_crop, (img_size, img_size))
                face_rgb     = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                face_arr     = face_rgb.astype("float32") / 255.0
                face_arr     = np.reshape(face_arr, (1, img_size, img_size, 3))
                raw_pred     = model.predict(face_arr, verbose=0)[0][0]
                prediction_buffer.append(raw_pred)
                avg_pred     = sum(prediction_buffer) / len(prediction_buffer)
                mask_on      = avg_pred < 0.5
                conf         = float((1 - avg_pred) if mask_on else avg_pred)
                x1, y1, x2, y2 = fx, fy, fx+fw, fy+fh

            # ── Other models use padded crop ──────────────────
            else:
                pad_x = int(fw * 0.3)
                pad_y = int(fh * 0.3)
                x1 = max(0, fx - pad_x);      y1 = max(0, fy - pad_y)
                x2 = min(w, fx + fw + pad_x); y2 = min(h, fy + fh + pad_y)
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                face_resized = cv2.resize(face_crop, (img_size, img_size))
                face_arr     = np.expand_dims(face_resized / 255.0, axis=0)
                raw_pred     = model.predict(face_arr, verbose=0)[0][0]
                prediction_buffer.append(raw_pred)
                avg_pred     = sum(prediction_buffer) / len(prediction_buffer)
                mask_on      = avg_pred < 0.5
                conf         = float((1 - avg_pred) if mask_on else avg_pred)

            label = "With Mask" if mask_on else "No Mask"
            color = (46, 204, 113) if mask_on else (231, 76, 60)

            # Only show if confident
            if conf >= 0.75:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                txt = f"{label} {conf*100:.1f}%"
                cv2.rectangle(frame, (x1, y1-30), (x1+len(txt)*11, y1), color, -1)
                cv2.putText(frame, txt, (x1+4, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                # Confidence bar
                bar_x, bar_y = x1, y2 + 10
                bar_w = int(conf * 150)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x+150, bar_y+10), (200,200,200), 1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+10), color, -1)

        # Model name overlay
        cv2.putText(frame, f"Model: {model_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, "Press Q to next model", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(f"Testing: {model_name}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    # Reset buffer for next model
    prediction_buffer.clear()
    print(f"{model_name} webcam test done!")

# ══════════════════════════════════════════════════════════════
# MAIN — Run all models
# ══════════════════════════════════════════════════════════════
print("=" * 55)
print("   MASK DETECTION — ALL MODELS TESTER")
print("=" * 55)

for model_name, (model_path, img_size) in MODELS.items():
    print(f"\n{'='*55}")
    print(f"  Loading: {model_name}")
    print(f"{'='*55}")

    model = load_model(model_path)
    print(f"Loaded! Params: {model.count_params():,}")

    # Part 1 — Sample images
    test_on_images(model_name, model, img_size)

    # Part 2 — Webcam
    input(f"\nPress ENTER to start WEBCAM test for {model_name}...")
    test_on_webcam(model_name, model, img_size)

    print(f"\n✅ {model_name} testing complete!")
    input("Press ENTER to test next model...\n")

print("\n" + "="*55)
print("   ALL MODELS TESTED SUCCESSFULLY!")
print("="*55)
