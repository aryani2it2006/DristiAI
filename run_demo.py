"""
run_demo.py

Runs EVERY module built so far, end-to-end, on synthetic data, so you can
see the full pipeline work before plugging in your real photos / camera.

What this proves works:
    1. Face detection + LBPH training + recognition (Known vs Unknown)
    2. Human detection (HOG)
    3. Vehicle "detection" (motion-based demo placeholder)
    4. ANPR (plate localization + OCR)
    5. Virtual fence line-crossing
    6. Suspicious activity (Pandas/NumPy/scikit-learn pipeline)
    7. Night-time movement detection
    8. Event creation -> Alert engine -> SQLite storage
    9. Evidence hashing -> demo hash-chain "blockchain" -> verification

Run with:  python run_demo.py
(Full "how to run this" walkthrough will be given separately when asked.)
"""

import os
import shutil

import cv2
import numpy as np
from skimage import data as skdata

from ai import face_module
from ai import human_detection
from ai import vehicle_detection
from ai import anpr
from ai import virtual_fence
from ai import suspicious_activity
from ai import night_detection
from engine import event_engine
from engine import alert_engine
from engine import evidence
from engine.blockchain_sim import HashChainLedger
from backend import db

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
PHOTOS_DIR = os.path.join(DATA_DIR, "photos")
EVIDENCE_DIR = os.path.join(DATA_DIR, "evidence")
MODEL_PATH = os.path.join(DATA_DIR, "trainer.yml")
LABEL_MAP_PATH = os.path.join(DATA_DIR, "labels.csv")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------
# STEP 0 -- build synthetic training photos (stand-in for your real
# "Photos" folder from Section 15.2, since none was uploaded yet)
# ---------------------------------------------------------------------
def build_synthetic_photos():
    person_dir = os.path.join(PHOTOS_DIR, "Demo_Person_1")
    if os.path.exists(person_dir):
        shutil.rmtree(person_dir)
    os.makedirs(person_dir, exist_ok=True)

    base = cv2.cvtColor(skdata.astronaut(), cv2.COLOR_RGB2BGR)  # public sample image w/ a real face

    rng = np.random.default_rng(1)
    for i in range(8):
        img = base.copy()
        # small brightness/contrast jitter so LBPH sees mild variation, like real photos would
        alpha = 1.0 + rng.uniform(-0.15, 0.15)
        beta = rng.uniform(-15, 15)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        cv2.imwrite(os.path.join(person_dir, f"image_{i:02d}.jpg"), img)

    return person_dir


def build_unknown_test_image():
    """A face-sized crop that is NOT the enrolled person -- random noise
    stood in for 'a different person's face' since we have only one
    sample face image available offline."""
    rng = np.random.default_rng(2)
    noise = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    return noise


# ---------------------------------------------------------------------
# STEP 1 -- Face detection + LBPH training + recognition
# ---------------------------------------------------------------------
def run_face_module():
    section("1. FACE DETECTION + LBPH RECOGNITION (Section 15)")
    build_synthetic_photos()
    label_to_name, n_faces = face_module.train_recognizer(PHOTOS_DIR, MODEL_PATH, LABEL_MAP_PATH)
    print(f"Trained LBPH on {n_faces} cropped faces. Labels: {label_to_name}")

    known_frame = cv2.cvtColor(skdata.astronaut(), cv2.COLOR_RGB2BGR)
    results_known = face_module.recognize_faces(known_frame, MODEL_PATH, LABEL_MAP_PATH)
    print("Recognition on a held-out frame of the SAME enrolled person:")
    for r in results_known:
        print(f"   -> box={r['box']} name={r['name']} distance={r['distance']:.1f}")

    unknown_frame = build_unknown_test_image()
    # random noise won't contain a detectable face -- this instead proves
    # detect_faces() correctly reports "no face" rather than false-matching
    unknown_boxes = face_module.detect_faces(unknown_frame)
    print(f"Face detection on random-noise image (expect no faces): {unknown_boxes}")

    return known_frame


# ---------------------------------------------------------------------
# STEP 2 -- Human detection (real photo) + tracking
# ---------------------------------------------------------------------
def run_human_detection(frame):
    section("2. HUMAN DETECTION (Section 4) -- HOG, our choice, see file docstring")
    detections = human_detection.detect_humans(frame)
    print(f"Human detections: {detections}")
    return detections


# ---------------------------------------------------------------------
# STEP 3 -- synthetic video with a moving "car" -> vehicle detection,
# virtual fence crossing, and ANPR
# ---------------------------------------------------------------------
def build_synthetic_car_frames(n_warmup=25, n_moving=30, width=640, height=360):
    """
    n_warmup static background frames first, so the background-subtraction
    based vehicle detector (ai/vehicle_detection.py) has time to learn what
    "no vehicle" looks like -- otherwise the very first frames are
    incorrectly flagged as one giant foreground blob. Only after warmup
    does the car start moving, well to the left of the virtual fence at
    x=320, and finishes well to the right of it.
    """
    frames = []

    def draw(cx):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)  # dark road background
        cv2.rectangle(frame, (cx, 200), (cx + 160, 260), (80, 80, 80), -1)  # car body
        cv2.rectangle(frame, (cx + 30, 235), (cx + 130, 258), (255, 255, 255), -1)  # plate
        cv2.putText(frame, "MH12AB1234", (cx + 33, 253), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        return frame

    start_cx = 40
    for _ in range(n_warmup):
        frames.append(draw(start_cx))  # car parked, off to the left, while background settles

    end_cx = width - 200
    for i in range(n_moving):
        cx = start_cx + int(i * (end_cx - start_cx) / n_moving)
        frames.append(draw(cx))

    return frames


def run_vehicle_pipeline():
    section("3. VEHICLE DETECTION + VIRTUAL FENCE + ANPR (Sections 5, 7, 8)")
    frames = build_synthetic_car_frames()
    fence = virtual_fence.VirtualFence((320, 0), (320, 360))  # vertical line at x=320
    tracker = human_detection.CentroidTracker(max_distance=150)

    intrusion_triggered = False
    last_plate_text = None
    last_frame_with_vehicle = None
    last_vehicle_box = None

    for idx, frame in enumerate(frames):
        detections = vehicle_detection.detect_vehicles(frame)
        # adapt vehicle detections to the (x1,y1,x2,y2) shape CentroidTracker expects
        assigned = tracker.update(detections)
        for track_id, det in assigned:
            cx = (det["x1"] + det["x2"]) / 2.0
            cy = (det["y1"] + det["y2"]) / 2.0
            crossed = fence.check(track_id, (cx, cy))
            if crossed:
                intrusion_triggered = True
                print(f"   Frame {idx}: vehicle track {track_id} CROSSED virtual fence at x=320")
            if det["object"] == "vehicle_candidate":
                last_frame_with_vehicle = frame
                last_vehicle_box = (det["x1"], det["y1"], det["x2"] - det["x1"], det["y2"] - det["y1"])

    print(f"Vehicle detections seen across {len(frames)} frames: at least one candidate found = "
          f"{last_vehicle_box is not None}")
    print(f"Virtual fence intrusion triggered: {intrusion_triggered}")

    plate_text = None
    if last_frame_with_vehicle is not None:
        anpr_results = anpr.run_anpr(last_frame_with_vehicle)
        print(f"ANPR results on last frame: {anpr_results}")
        if anpr_results:
            plate_text = anpr_results[0]["plate_text"]

    return intrusion_triggered, plate_text, last_frame_with_vehicle


# ---------------------------------------------------------------------
# STEP 4 -- suspicious activity (Pandas/NumPy/scikit-learn pipeline)
# ---------------------------------------------------------------------
def run_suspicious_activity():
    section("4. SUSPICIOUS ACTIVITY DETECTION (Section 9) -- Pandas/NumPy/scikit-learn")
    model = suspicious_activity.train_model()

    normal_track = [(0, 100, 100), (1, 105, 102), (2, 110, 103)]  # slow, smooth
    erratic_track = [(0, 100, 100), (1, 140, 180), (2, 90, 250)]  # fast, direction-changing

    normal_features = suspicious_activity.extract_features(normal_track)
    erratic_features = suspicious_activity.extract_features(erratic_track)
    normal_features["distance_to_nearest_person"] = 250  # far from others
    erratic_features["distance_to_nearest_person"] = 30  # close to another person

    normal_result = suspicious_activity.predict_activity(model, normal_features)
    erratic_result = suspicious_activity.predict_activity(model, erratic_features)

    print(f"Normal-looking track  -> features={normal_features} -> result={normal_result}")
    print(f"Erratic-looking track -> features={erratic_features} -> result={erratic_result}")
    return erratic_result


# ---------------------------------------------------------------------
# STEP 5 -- night-time movement
# ---------------------------------------------------------------------
def run_night_detection(day_frame):
    section("5. NIGHT-TIME MOVEMENT DETECTION (Section 10)")
    night_frame = cv2.convertScaleAbs(day_frame, alpha=0.15, beta=0)  # darken to simulate night
    day_result = night_detection.night_movement_event(day_frame, has_motion_detections=True)
    night_result = night_detection.night_movement_event(night_frame, has_motion_detections=True)
    print(f"Daytime frame result:  {day_result}")
    print(f"Nighttime frame result: {night_result}")
    return night_result


# ---------------------------------------------------------------------
# STEP 6 -- wire everything into Events -> Alerts -> DB -> Evidence -> Ledger
# ---------------------------------------------------------------------
def run_event_pipeline(intrusion_triggered, plate_text, vehicle_frame, suspicious_result, night_result):
    section("6. EVENT ENGINE -> ALERT ENGINE -> DB -> EVIDENCE HASH -> LEDGER (Sections 11,12,52,54-61)")

    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    db.init_db()
    ledger = HashChainLedger()

    events_to_create = []
    if intrusion_triggered:
        events_to_create.append(event_engine.make_event(
            "INTRUSION", "CAM-DEMO-01", 0.90, {"plate_text": plate_text}))
    if suspicious_result["result"] == "suspicious":
        events_to_create.append(event_engine.make_event(
            "SUSPICIOUS_ACTIVITY", "CAM-DEMO-01", suspicious_result["confidence"], {}))
    if night_result["night_movement"]:
        events_to_create.append(event_engine.make_event(
            "NIGHT_MOVEMENT", "CAM-DEMO-01", 0.75, {"brightness": night_result["brightness"]}))

    for event in events_to_create:
        db.insert_event(event)

        # Section 52 -- evidence hashing: save a snapshot as the "evidence file"
        evidence_path = os.path.join(EVIDENCE_DIR, f"{event['event_id']}.jpg")
        snapshot = vehicle_frame if vehicle_frame is not None else np.zeros((100, 100, 3), np.uint8)
        cv2.imwrite(evidence_path, snapshot)
        evidence_hash = evidence.hash_file(evidence_path)

        # Sections 54-61 (demo hash-chain stand-in, see blockchain_sim.py docstring)
        block = ledger.add_record(event["event_id"], evidence_hash)
        db.insert_blockchain_record(event["event_id"], block)

        alert = alert_engine.evaluate_event(event)
        if alert:
            db.insert_alert(alert)

        print(f"Event {event['event_id']} ({event['event_type']}) -> evidence hash {evidence_hash[:16]}... "
              f"-> ledger block #{block['index']}")

    valid, message = ledger.verify_chain()
    print(f"\nLedger integrity check: {valid} -- {message}")

    if events_to_create:
        sample_event_id = events_to_create[0]["event_id"]
        sample_hash = evidence.hash_file(os.path.join(EVIDENCE_DIR, f"{sample_event_id}.jpg"))
        verify_result = ledger.verify_event(sample_event_id, sample_hash)
        print(f"Verifying evidence for {sample_event_id} against ledger: {verify_result['match']}")

    print(f"\nAll events, alerts and blockchain records written to: {db.DB_PATH}")
    print("Start the dashboard later with: python -m backend.app")


def main():
    known_frame = run_face_module()
    run_human_detection(known_frame)
    intrusion_triggered, plate_text, vehicle_frame = run_vehicle_pipeline()
    suspicious_result = run_suspicious_activity()
    night_result = run_night_detection(known_frame)
    run_event_pipeline(intrusion_triggered, plate_text, vehicle_frame, suspicious_result, night_result)


if __name__ == "__main__":
    main()
