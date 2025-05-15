from flask import Flask, Response, send_file
import cv2
import pickle
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import pandas as pd
import time
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load cosine KNN model
with open("exercise_cosine_model.pkl", "rb") as f:
    model_data = pickle.load(f)
    X_train = model_data["X"]
    y_train = model_data["y"]

# Setup pose estimator
w, h = model_wh('640x480')
e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))
cam = cv2.VideoCapture(1)

# Keypoints index used in flat vector (first 18)
def extract_flat_vector(human, image_w, image_h):
    features = []
    for i in range(18):
        if i in human.body_parts:
            bp = human.body_parts[i]
            x, y = bp.x * image_w, bp.y * image_h
        else:
            x, y = 0.0, 0.0
        features.extend([x, y])
    return features

def predict_pose(flat_vector):
    input_vector = np.array(flat_vector).reshape(1, -1)
    # Normalize the input vector
    input_norm = input_vector / (np.linalg.norm(input_vector) + 1e-6)
    # Use cosine_similarity instead of dot product
    similarity_scores = cosine_similarity(input_norm, X_train).flatten()
    top_k = np.argsort(similarity_scores)[-5:]
    top_k_labels = y_train[top_k]
    return pd.Series(top_k_labels).value_counts().idxmax()

def gen_frames():
    global e
    fps_time = time.time()
    while True:
        success, frame = cam.read()
        if not success:
            break

        humans = e.inference(frame, resize_to_default=True, upsample_size=4.0)
        frame = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)

        if humans:
            try:
                flat_vector = extract_flat_vector(humans[0], frame.shape[1], frame.shape[0])
                prediction = predict_pose(flat_vector)
                cv2.putText(frame, f"Pose: {prediction}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            except Exception as e:
                print("Prediction error:", e)

        fps = 1.0 / (time.time() - fps_time)
        cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 105, 180), 2)
        fps_time = time.time()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return send_file("frontend.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)