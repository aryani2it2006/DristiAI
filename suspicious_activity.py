"""
ai/suspicious_activity.py

Implements Module G (Section 9) -- Suspicious Activity Detection.

Section 9.2/9.3/9.4/9.5 explicitly names the technologies to use:
    Pandas       -> organize extracted features into a tabular dataset
    NumPy        -> numerical operations on those features
    Scikit-learn -> the actual ML classification model

Pipeline (Section 9.1):
    Human Detection -> Tracking/Movement -> Feature Extraction ->
    Activity Analysis -> Suspicious / Normal Result

SPEC GAP: the spec does not name which scikit-learn algorithm, nor
provide real labeled training data. DEMO CHOICE (flagged): a
RandomForestClassifier trained on a small synthetic dataset built from a
simple, explicit rule (fast + erratic movement = suspicious). This is
only to prove the Pandas -> NumPy -> scikit-learn pipeline runs
end-to-end. It must be retrained on real, reviewed activity data before
being trusted for anything real -- confirm the labeling process before
that happens.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURE_COLUMNS = ["speed", "direction_change", "distance_to_nearest_person"]


def build_synthetic_training_set(n_samples=400, random_state=0):
    """
    Builds a synthetic labeled dataset purely so the classifier has
    something to train on for this demo. Rule used to generate labels
    (OUR rule, not from the spec): high speed + high direction change +
    close proximity to another person => suspicious.
    """
    rng = np.random.default_rng(random_state)
    speed = rng.uniform(0, 40, n_samples)                     # pixels/frame
    direction_change = rng.uniform(0, 180, n_samples)         # degrees
    distance = rng.uniform(5, 300, n_samples)                 # pixels

    df = pd.DataFrame({
        "speed": speed,
        "direction_change": direction_change,
        "distance_to_nearest_person": distance,
    })

    score = (
        (df["speed"] > 20).astype(int)
        + (df["direction_change"] > 90).astype(int)
        + (df["distance_to_nearest_person"] < 60).astype(int)
    )
    df["label"] = (score >= 2).astype(int)  # 1 = suspicious, 0 = normal
    return df


def train_model():
    df = build_synthetic_training_set()
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["label"].to_numpy()
    model = RandomForestClassifier(n_estimators=50, random_state=0)
    model.fit(X, y)
    return model


def extract_features(track_history):
    """
    track_history: list of (frame_idx, x, y) centroids for ONE tracked
    person, oldest first. Returns a feature dict per Section 9.2/9.3.
    """
    if len(track_history) < 2:
        return {"speed": 0.0, "direction_change": 0.0, "distance_to_nearest_person": 999.0}

    (_, x0, y0), (_, x1, y1) = track_history[-2], track_history[-1]
    speed = float(np.hypot(x1 - x0, y1 - y0))

    direction_change = 0.0
    if len(track_history) >= 3:
        (_, xa, ya), (_, xb, yb), (_, xc, yc) = track_history[-3:]
        v1 = np.array([xb - xa, yb - ya])
        v2 = np.array([xc - xb, yc - yb])
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            direction_change = float(np.degrees(np.arccos(cos_angle)))

    return {"speed": speed, "direction_change": direction_change,
            "distance_to_nearest_person": 999.0}  # filled in by caller if multiple tracks exist


def predict_activity(model, feature_dict):
    X = pd.DataFrame([feature_dict])[FEATURE_COLUMNS].to_numpy()
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]
    return {"result": "suspicious" if pred == 1 else "normal", "confidence": float(proba)}
