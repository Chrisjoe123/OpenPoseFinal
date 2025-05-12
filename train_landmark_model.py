# train_landmark_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Fungsi bantu
def get_vector(a, b):
    return np.array([b[0] - a[0], b[1] - a[1]])

def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
    return dot / norm

# Langkah 1: Baca file CSV
print("[INFO] Membaca dataset landmarks dan label...")
landmarks = pd.read_csv("Dataset/landmarks.csv")
labels = pd.read_csv("Dataset/labels.csv")

# Langkah 2: Tambahkan label ke setiap frame berdasarkan vid_id
print("[INFO] Menambahkan label berdasarkan vid_id...")
label_map = labels.set_index("vid_id")["class"].to_dict()
landmarks["class"] = landmarks["vid_id"].map(label_map)

# Langkah 3: Drop kolom non-fitur
df = landmarks.drop(columns=["vid_id", "frame_order"])

# Langkah 4: Bersihkan data dari NaN / inf
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Langkah 5: Pisahkan fitur (X) dan label (y)
X = df.drop(columns=["class"])
y = df["class"]

# Langkah 6: Tambahkan fitur cosine dari segmen tubuh
print("[INFO] Menambahkan fitur cosine dari vektor tubuh...")
cosine_right_arm, cosine_left_arm = [], []
cosine_right_leg, cosine_left_leg = [], []

for i in range(len(X)):
    kp = lambda name: (X[f"x_{name}"].iloc[i], X[f"y_{name}"].iloc[i])

    def safe_cos(p1a, p1b, p2a, p2b):
        try:
            return cosine_similarity(get_vector(kp(p1a), kp(p1b)), get_vector(kp(p2a), kp(p2b)))
        except:
            return 0.0

    cosine_right_arm.append(safe_cos("right_shoulder", "right_elbow", "right_elbow", "right_wrist"))
    cosine_left_arm.append(safe_cos("left_shoulder", "left_elbow", "left_elbow", "left_wrist"))
    cosine_right_leg.append(safe_cos("right_hip", "right_knee", "right_knee", "right_ankle"))
    cosine_left_leg.append(safe_cos("left_hip", "left_knee", "left_knee", "left_ankle"))

X["cosine_right_arm"] = cosine_right_arm
X["cosine_left_arm"] = cosine_left_arm
X["cosine_right_leg"] = cosine_right_leg
X["cosine_left_leg"] = cosine_left_leg

# Langkah 7: Split data
print("[INFO] Membagi data menjadi train/test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Langkah 8: Latih model
print("[INFO] Melatih model RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Langkah 9: Evaluasi
print("[INFO] Evaluasi model...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Akurasi: %.2f%%" % (accuracy_score(y_test, y_pred) * 100))

# Langkah 10: Simpan model
print("[INFO] Menyimpan model ke exercise_landmark_model.pkl...")
joblib.dump(model, "exercise_landmark_model.pkl")
print("[SELESAI] Model berhasil disimpan.")
