import argparse
import logging
import time
import os
import csv
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from collections import deque
import joblib

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-CosineLandmarkOnly')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

keypoint_names = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky_1", "right_pinky_1",
    "left_index_1", "right_index_1", "left_thumb_2", "right_thumb_2",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index"
]

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def extract_features_from_human(human, image_w, image_h):
    features = {}
    keypoints = {}
    for i, name in enumerate(keypoint_names):
        if i in human.body_parts:
            bp = human.body_parts[i]
            x, y = int(bp.x * image_w), int(bp.y * image_h)
            keypoints[name] = (x, y, 0.0)
            features[f"x_{name}"] = x
            features[f"y_{name}"] = y
            features[f"z_{name}"] = 0.0
        else:
            features[f"x_{name}"] = 0.0
            features[f"y_{name}"] = 0.0
            features[f"z_{name}"] = 0.0

    def vec(p1, p2):
        return np.array([p2[0] - p1[0], p2[1] - p1[1]])

    try:
        v1 = vec(keypoints["right_shoulder"], keypoints["right_elbow"])
        v2 = vec(keypoints["right_elbow"], keypoints["right_wrist"])
        features["cosine_right_arm"] = cosine_similarity(v1, v2)
    except:
        features["cosine_right_arm"] = 0.0

    try:
        v1 = vec(keypoints["left_shoulder"], keypoints["left_elbow"])
        v2 = vec(keypoints["left_elbow"], keypoints["left_wrist"])
        features["cosine_left_arm"] = cosine_similarity(v1, v2)
    except:
        features["cosine_left_arm"] = 0.0

    try:
        v1 = vec(keypoints["right_hip"], keypoints["right_knee"])
        v2 = vec(keypoints["right_knee"], keypoints["right_ankle"])
        features["cosine_right_leg"] = cosine_similarity(v1, v2)
    except:
        features["cosine_right_leg"] = 0.0

    try:
        v1 = vec(keypoints["left_hip"], keypoints["left_knee"])
        v2 = vec(keypoints["left_knee"], keypoints["left_ankle"])
        features["cosine_left_leg"] = cosine_similarity(v1, v2)
    except:
        features["cosine_left_leg"] = 0.0

    try:
        v1 = vec(keypoints["left_shoulder"], keypoints["left_hip"])
        v2 = vec(keypoints["right_shoulder"], keypoints["right_hip"])
        features["cosine_torso"] = cosine_similarity(v1, v2)
    except:
        features["cosine_torso"] = 0.0
 
    try:
        v1 = vec(keypoints["left_hip"], keypoints["left_shoulder"])
        v2 = vec(keypoints["left_shoulder"], keypoints["nose"])
        features["cosine_core"] = cosine_similarity(v1, v2)
    except:
        features["cosine_core"] = 0.0

    return features

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation with ML classifier (landmark + cosine only)')
    parser.add_argument('--camera', type=int, default=1)
    parser.add_argument('--resize', type=str, default='320x240')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0)
    parser.add_argument('--model', type=str, default='mobilenet_thin')
    parser.add_argument('--ml-model-path', type=str, default='exercise_landmark_model.pkl')
    parser.add_argument('--log-predictions', type=str, default='predicted_log.csv')
    args = parser.parse_args()

    logger.info('Loading ML model...')
    model = joblib.load(args.ml_model_path)

    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()

    if not ret_val:
        logger.error("Tidak bisa membaca kamera/video.")
        exit()

    # Inisialisasi file log CSV
    log_file = open(args.log_predictions, mode='w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['timestamp', 'frame_id', 'predicted_label'])
    frame_id = 0

    while True:
        ret_val, image = cam.read()
        if not ret_val:
            break

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        if humans:
            features = extract_features_from_human(humans[0], image.shape[1], image.shape[0])
            try:
                df = pd.DataFrame([features])
                prediction = model.predict(df)[0]
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_writer.writerow([timestamp, frame_id, prediction])
                frame_id += 1

                cv2.putText(image,
                            f"Exercise: {prediction}",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 0), 2)
            except Exception as e:
                logger.error(f"Prediction error: {e}")

        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 105, 180), 2)

        cv2.imshow('tf-pose-estimation with ML', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    log_file.close()
    cv2.destroyAllWindows()
