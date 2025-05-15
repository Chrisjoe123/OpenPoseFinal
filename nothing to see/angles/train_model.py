import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load datasets
angles_df = pd.read_csv("Dataset/angles.csv")
labels_df = pd.read_csv("Dataset/labels.csv")

# Gabungkan berdasarkan vid_id dan frame_order
merged_df = pd.merge(angles_df, labels_df, on="vid_id")

# Hapus baris yang memiliki NaN
merged_df = merged_df.dropna()

# Pisahkan fitur dan label
X = merged_df.drop(columns=['vid_id', 'frame_order', 'class'])
y = merged_df['class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Simpan model
joblib.dump(model, "exercise_classifier.pkl")

# Uji akurasi
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
