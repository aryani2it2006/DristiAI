"""
ai/face_module.py

Implements Section 15 of the IBVAP spec:
    - Haar Cascade face DETECTION  (6.1, 6.3)
    - LBPH face RECOGNITION        (6.4, 6.5, 6.6)

Spec-mandated technology: OpenCV, Haar Cascade, LBPH, grayscale images,
trainer.yml. Nothing else is introduced here.

DEMO NOTE: This file is spec-accurate. What's synthetic is only the DATA
fed into it (run_demo.py builds fake training photos from a public sample
image, since no real "Photos" folder was provided yet). The moment you
point PHOTOS_DIR at your real photos, this same code trains on real data.
"""

import os
import cv2
import numpy as np

# Bundled with opencv-contrib-python -- this is the standard, unmodified
# haarcascade_frontalface_default.xml referenced in Section 15.2.
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# LBPH prediction returns a "distance" -- lower means closer match.
# The spec (6.4 / 15.6) says a threshold decides Known vs Unknown but does
# not give a numeric value. 70 is a commonly used LBPH default; flagged
# here as OUR choice, not a spec value. Tune this once you have real data.
LBPH_DISTANCE_THRESHOLD = 70


def _face_cascade():
    return cv2.CascadeClassifier(CASCADE_PATH)


def detect_faces(bgr_frame):
    """
    Section 6.1 / 6.3 -- Haar Cascade face detection only.
    Returns a list of (x, y, w, h) boxes. Does NOT identify anyone.
    """
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    cascade = _face_cascade()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return list(faces)


def train_recognizer(photos_dir, model_path, label_map_path):
    """
    Section 15.3 / 15.4 -- Face Cropping + Training Model.

    Expects:
        photos_dir/
            <PersonName>/
                image_01.jpg
                image_02.jpg
                ...

    Workflow (exactly as Section 15.3/15.5 describes):
        Read Image -> Grayscale -> Haar Cascade -> Detect Face ->
        Crop Face -> Assign Label -> Train LBPH -> Save trainer.yml
    """
    cascade = _face_cascade()
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces_for_training = []
    labels = []
    label_to_name = {}
    name_to_label = {}
    next_label = 0

    person_names = sorted(
        d for d in os.listdir(photos_dir) if os.path.isdir(os.path.join(photos_dir, d))
    )

    for person_name in person_names:
        if person_name not in name_to_label:
            name_to_label[person_name] = next_label
            label_to_name[next_label] = person_name
            next_label += 1
        label = name_to_label[person_name]

        person_dir = os.path.join(photos_dir, person_name)
        for fname in sorted(os.listdir(person_dir)):
            fpath = os.path.join(person_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            if len(faces) == 0:
                continue
            # largest detected face in the photo
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            cropped = gray[y:y + h, x:x + w]
            cropped = cv2.resize(cropped, (200, 200))
            faces_for_training.append(cropped)
            labels.append(label)

    if not faces_for_training:
        raise RuntimeError(
            "No faces could be detected in any training photo. "
            "Check that photos_dir contains clear, front-facing face images."
        )

    recognizer.train(faces_for_training, np.array(labels))
    recognizer.save(model_path)

    with open(label_map_path, "w") as f:
        for label, name in label_to_name.items():
            f.write(f"{label},{name}\n")

    return label_to_name, len(faces_for_training)


def _load_label_map(label_map_path):
    label_to_name = {}
    with open(label_map_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label_str, name = line.split(",", 1)
            label_to_name[int(label_str)] = name
    return label_to_name


def recognize_faces(bgr_frame, model_path, label_map_path,
                     threshold=LBPH_DISTANCE_THRESHOLD):
    """
    Section 15.5 / 15.6 -- Live Recognition + Recognition Logic.

    Returns a list of dicts:
        {"box": (x,y,w,h), "name": "<Person>" or "Unknown", "distance": float}
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    label_to_name = _load_label_map(label_map_path)

    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    cascade = _face_cascade()
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

    results = []
    for (x, y, w, h) in faces:
        cropped = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
        label, distance = recognizer.predict(cropped)
        if distance <= threshold:
            name = label_to_name.get(label, "Unknown")
        else:
            name = "Unknown"
        results.append({"box": (int(x), int(y), int(w), int(h)),
                         "name": name,
                         "distance": float(distance)})
    return results
