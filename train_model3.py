import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# Load landmarks.csv and labels.csv
landmarks = pd.read_csv("Dataset/landmarks.csv")
labels = pd.read_csv("Dataset/labels.csv")

# Merge to add class to each frame
data = pd.merge(landmarks, labels, on="vid_id")

# Define the 18 keypoints to keep (OpenPose standard order index)
keypoints = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
    "left_eye", "right_ear", "left_ear"
]

# Build flattened vector for each row
X = []
for _, row in data.iterrows():
    vector = []
    for kpt in keypoints:
        vector.append(row.get(f"x_{kpt}", 0.0))
        vector.append(row.get(f"y_{kpt}", 0.0))
    X.append(vector)

X = normalize(X, axis=1)
y = data["class"].values

# Test cosine similarity to confirm functionality
if len(X) > 1:
    test_similarity = cosine_similarity([X[0]], [X[1]])[0][0]
    print(f"Test cosine similarity between first two samples: {test_similarity}")

# Save normalized vectors and labels
with open("exercise_cosine_model.pkl", "wb") as f:
    pickle.dump({"X": X, "y": y}, f)

print("✅ Model trained and saved as exercise_cosine_model.pkl with 36 flat features.")