# Update of run_webcam.py to use ML model instead of manual rule-based classification

import argparse
import logging
import time
import os

import cv2
import numpy as np
import pandas as pd
from collections import deque
import joblib

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

# Setup logger for debugging output
logger = logging.getLogger('TfPoseEstimator-WebCam-ML')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0  # For measuring frame-per-second

# Menghitung sudut antara 3 titik (menggunakan rumus cosine)
def calculate_angle(a, b, c):
    # Vector BA
    ba_x = a[0] - b[0]
    ba_y = a[1] - b[1]
    # Vector BC
    bc_x = c[0] - b[0]
    bc_y = c[1] - b[1]
    
    # Cosine of the angle between BA and BC
    cosine_angle = (ba_x * bc_x + ba_y * bc_y) / (
        np.sqrt(ba_x**2 + ba_y**2) * np.sqrt(bc_x**2 + bc_y**2) + 1e-6)
    
    # Angle in degrees
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) * 180.0 / np.pi
    
    return angle

# Mengekstrak sudut tubuh dari hasil keypoints tf-pose-estimation
def extract_body_angles(humans):
    """Extract key body angles from pose estimation results"""
    if not humans or len(humans) == 0:
        return None

    human = humans[0]
    body_parts = human.body_parts
    
    # Dictionary to store the angles
    angles = {}
    
    # Mapping of indices to body parts
    """
    {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
     5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
     10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
     15: "LEye", 16: "REar", 17: "LEar"}
    
    Note: This mapping might be different for your model, adjust as needed.
    """
    
    # We'll extract all key points first
    keypoints = {}

    # Simpan semua keypoint ke dalam dictionary (index: (x, y))
    for i, body_part in body_parts.items():
        keypoints[i] = (body_part.x, body_part.y)

    # Hitung titik tengah bahu dan pinggul
    if 5 in keypoints and 2 in keypoints:
        keypoints['mid_shoulder'] = ((keypoints[5][0] + keypoints[2][0])/2,
                                    (keypoints[5][1] + keypoints[2][1])/2)
    if 11 in keypoints and 8 in keypoints:
        keypoints['mid_hip'] = ((keypoints[11][0] + keypoints[8][0])/2,
                              (keypoints[11][1] + keypoints[8][1])/2)

    # Hitung sudut-sudut penting
    if 2 in keypoints and 3 in keypoints and 8 in keypoints:
        angles['right_elbow_right_shoulder_right_hip'] = calculate_angle(keypoints[3], keypoints[2], keypoints[8])
    if 5 in keypoints and 6 in keypoints and 11 in keypoints:
        angles['left_elbow_left_shoulder_left_hip'] = calculate_angle(keypoints[6], keypoints[5], keypoints[11])
    
    if 9 in keypoints and 'mid_hip' in keypoints and 12 in keypoints:
        angles['right_knee_mid_hip_left_knee'] = calculate_angle(keypoints[9], keypoints['mid_hip'], keypoints[12])
    
    if 8 in keypoints and 9 in keypoints and 10 in keypoints:
        angles['right_hip_right_knee_right_ankle'] = calculate_angle(keypoints[8], keypoints[9], keypoints[10])
    
    if 11 in keypoints and 12 in keypoints and 13 in keypoints:
        angles['left_hip_left_knee_left_ankle'] = calculate_angle(keypoints[11], keypoints[12], keypoints[13])
    
    if 4 in keypoints and 3 in keypoints and 2 in keypoints:
        angles['right_wrist_right_elbow_right_shoulder'] = calculate_angle(keypoints[4], keypoints[3], keypoints[2])
        
    if 7 in keypoints and 6 in keypoints and 5 in keypoints:
        angles['left_wrist_left_elbow_left_shoulder'] = calculate_angle(keypoints[7], keypoints[6], keypoints[5])

    return angles

# Klasifikasi gerakan berdasarkan sudut dengan model ML
def classify_with_model(angles, model):
    if not angles:
        return "Unknown"
    try:
        angle_vector = pd.DataFrame([angles])  # Buat 1 baris dataframe dari dictionary sudut
        prediction = model.predict(angle_vector)
        return prediction[0]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "Unknown"

if __name__ == '__main__':
    # Argument dari command-line
    parser = argparse.ArgumentParser(description='tf-pose-estimation with ML classifier')
    parser.add_argument('--camera', type=str, default=0)
    parser.add_argument('--resize', type=str, default='320x240',)
    parser.add_argument('--resize-out-ratio', type=float, default=4.0)
    parser.add_argument('--model', type=str, default='mobilenet_thin')
    parser.add_argument('--ml-model-path', type=str, default='exercise_classifier.pkl',
                        help='Path to the trained ML model')
    args = parser.parse_args()

    # Load model ML yang sudah ditraining
    logger.info('Loading ML model...')
    model = joblib.load(args.ml_model_path)

    # Inisialisasi pose estimator
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

    # Mulai video (webcam atau file)
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()

    while True:
        ret_val, image = cam.read()
        if not ret_val or image is None:
            print("Frame kosong. Mungkin video habis atau gagal dibaca.")
            break

        # Proses deteksi pose
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        if humans:
            # Hitung sudut dari pose
            angles = extract_body_angles(humans)

            # Klasifikasikan gerakan berdasarkan sudut menggunakan model ML
            exercise = classify_with_model(angles, model)

            # Tampilkan nama gerakan ke layar
            cv2.putText(image,
                        f"Exercise: {exercise}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 105, 180), 2)

        # Tampilkan FPS
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 105, 180), 2)

        # Tampilkan frame ke layar
        cv2.imshow('tf-pose-estimation with ML', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break  # ESC untuk keluar loop

    cv2.destroyAllWindows()
