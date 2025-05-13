from flask import Flask, Response, send_file
import cv2
import pandas as pd
import joblib
from datetime import datetime
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

# Function extract_features_from_human harus ada di file ini jika tidak di-import
from your_script_above import extract_features_from_human

app = Flask(__name__)

# Load model ML dan TfPoseEstimator
model = joblib.load('exercise_landmark_model.pkl')
w, h = model_wh('320x240')
e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))
cam = cv2.VideoCapture(0)  # Gunakan 1 jika 0 tidak bekerja

def generate():
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        humans = e.inference(frame, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
        frame = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)

        if humans:
            features = extract_features_from_human(humans[0], frame.shape[1], frame.shape[0])
            df = pd.DataFrame([features])
            prediction = model.predict(df)[0]
            cv2.putText(frame, f"Exercise: {prediction}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/')
def index():
    return send_file('frontend.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
