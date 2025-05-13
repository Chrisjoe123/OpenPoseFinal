from flask import Flask, Response, send_file
import cv2
import pandas as pd
import joblib
from datetime import datetime
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import time
# Function extract_features_from_human harus ada di file ini jika tidak di-import
from run_webcam_landmark import extract_features_from_human

app = Flask(__name__)

# Load model ML dan TfPoseEstimator
model = joblib.load('exercise_landmark_model.pkl')
w, h = model_wh('320x240')
 
# Harus diletakkan DI SINI sebelum generate()
e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))
cam = cv2.VideoCapture(1)
 
def generate():
    global e
    import time
    fps_time = time.time()
 
    while True:
        try:
            ret, frame = cam.read()
            if not ret:
                print("❌ Kamera tidak memberikan frame.")
                break
 
            humans = e.inference(frame, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
            frame = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)
 
            if humans:
                features = extract_features_from_human(humans[0], frame.shape[1], frame.shape[0])
                df = pd.DataFrame([features])
                prediction = model.predict(df)[0]
                print("✅ Prediksi:", prediction)
                cv2.putText(frame, f"Exercise: {prediction}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 0, 128), 2)
            else:
                print("⚠️ Tidak ada pose terdeteksi")
 
            # Tambahkan FPS info
            fps = 1.0 / (time.time() - fps_time)
            cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)
            fps_time = time.time()
 
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                print("❌ Gagal encode JPG")
                continue
 
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        except Exception as e:
            print("⚠️ Error dalam loop generate():", e)

@app.route('/')
def index():
    return send_file('frontend.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
