import argparse
import logging
import time
import os

import cv2
import numpy as np
import pandas as pd
from collections import deque

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

# Load datasets
def load_datasets(labels_path, landmarks_path):
    try:
        # Load labels dataset
        labels_df = pd.read_csv(labels_path)
        logger.info(f'Loaded labels dataset with {len(labels_df)} entries')
        
        # Load landmarks dataset
        landmarks_df = pd.read_csv(landmarks_path)
        logger.info(f'Loaded landmarks dataset with {len(landmarks_df)} entries')
        
        # Extract unique exercise classes
        exercise_classes = labels_df['class'].unique()
        logger.info(f'Found exercise classes: {exercise_classes}')
        
        return labels_df, landmarks_df, exercise_classes
    except Exception as e:
        logger.error(f'Error loading datasets: {e}')
        return None, None, None

# Extract landmark features for pose comparison/classification
def extract_pose_features(humans):
    features = []
    if len(humans) == 0:
        return None
    
    human = humans[0]  # Use the first detected person
    
    # Extract key body joint positions
    joints = {}
    if 0 in human.body_parts:  # Nose
        joints['nose'] = (human.body_parts[0].x, human.body_parts[0].y)
    if 1 in human.body_parts and 2 in human.body_parts:  # Eyes
        joints['eyes'] = ((human.body_parts[1].x + human.body_parts[2].x)/2, 
                         (human.body_parts[1].y + human.body_parts[2].y)/2)
    if 5 in human.body_parts and 6 in human.body_parts:  # Shoulders
        joints['shoulders'] = ((human.body_parts[5].x + human.body_parts[6].x)/2, 
                              (human.body_parts[5].y + human.body_parts[6].y)/2)
        joints['left_shoulder'] = (human.body_parts[5].x, human.body_parts[5].y)
        joints['right_shoulder'] = (human.body_parts[6].x, human.body_parts[6].y)
    if 7 in human.body_parts and 8 in human.body_parts:  # Elbows
        joints['left_elbow'] = (human.body_parts[7].x, human.body_parts[7].y)
        joints['right_elbow'] = (human.body_parts[8].x, human.body_parts[8].y)
    if 9 in human.body_parts and 10 in human.body_parts:  # Wrists
        joints['left_wrist'] = (human.body_parts[9].x, human.body_parts[9].y)
        joints['right_wrist'] = (human.body_parts[10].x, human.body_parts[10].y)
    if 11 in human.body_parts and 12 in human.body_parts:  # Hips
        joints['hips'] = ((human.body_parts[11].x + human.body_parts[12].x)/2, 
                         (human.body_parts[11].y + human.body_parts[12].y)/2)
    if 13 in human.body_parts and 14 in human.body_parts:  # Knees
        joints['left_knee'] = (human.body_parts[13].x, human.body_parts[13].y)
        joints['right_knee'] = (human.body_parts[14].x, human.body_parts[14].y)
    if 15 in human.body_parts and 16 in human.body_parts:  # Ankles
        joints['left_ankle'] = (human.body_parts[15].x, human.body_parts[15].y)
        joints['right_ankle'] = (human.body_parts[16].x, human.body_parts[16].y)
    
    # Calculate features from joints positions (if available)
    if 'shoulders' in joints and 'hips' in joints:
        # Torso length
        features.append(abs(joints['shoulders'][1] - joints['hips'][1]))
    
    if 'left_shoulder' in joints and 'left_elbow' in joints and 'left_wrist' in joints:
        # Left arm angles
        features.append(calculate_angle(joints['left_shoulder'], joints['left_elbow'], joints['left_wrist']))
    
    if 'right_shoulder' in joints and 'right_elbow' in joints and 'right_wrist' in joints:
        # Right arm angles
        features.append(calculate_angle(joints['right_shoulder'], joints['right_elbow'], joints['right_wrist']))
    
    if 'hips' in joints and 'left_knee' in joints and 'left_ankle' in joints:
        # Left leg angles
        features.append(calculate_angle(joints['hips'], joints['left_knee'], joints['left_ankle']))
    
    if 'hips' in joints and 'right_knee' in joints and 'right_ankle' in joints:
        # Right leg angles
        features.append(calculate_angle(joints['hips'], joints['right_knee'], joints['right_ankle']))
    
    return features if len(features) > 0 else None

# Calculate angle between three points
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

# Simple exercise classifier based on joint positions and angles
def classify_exercise(features, landmarks_df, labels_df):
    if features is None or len(features) < 4:
        return "Unknown"
    
    # This is a simplified example. In practice, you would use machine learning
    # based on the landmarks dataset to accurately classify exercises.
    
    # For demonstration, we'll use some basic rules based on body posture:
    arm_angle_avg = (features[1] + features[2]) / 2
    leg_angle_avg = (features[3] + features[4]) / 2
    
    if arm_angle_avg > 150:  # Arms extended
        if leg_angle_avg < 140:  # Legs bent
            return "squat"
        else:
            return "jumping_jack"
    elif 70 < arm_angle_avg < 120:  # Arms at medium angle
        if leg_angle_avg > 160:  # Legs straight
            return "push_up"
        else:
            return "situp"
    elif arm_angle_avg < 60:  # Arms bent significantly
        return "pull_up"
    
    return "Unknown"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=str, default=0)
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--labels', type=str, default='labels.csv',
                        help='path to the labels.csv file')
    parser.add_argument('--landmarks', type=str, default='landmarks.csv',
                        help='path to the landmarks.csv file')
    args = parser.parse_args()

    # Load datasets
    labels_df, landmarks_df, exercise_classes = load_datasets(args.labels, args.landmarks)

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    # For exercise classification stabilization
    pose_history = deque(maxlen=10)
    current_exercise = "Unknown"
    exercise_counter = 0
    exercise_started = False

    while True:
        ret_val, image = cam.read()

        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        # Extract features and classify exercise if humans detected
        if humans:
            features = extract_pose_features(humans)
            if features:
                detected_exercise = classify_exercise(features, landmarks_df, labels_df)
                pose_history.append(detected_exercise)
                
                # Stabilize classification by taking most frequent exercise in history
                if len(pose_history) >= 5:
                    # Count occurrences of each exercise type in history
                    exercise_counts = {}
                    for ex in pose_history:
                        if ex in exercise_counts:
                            exercise_counts[ex] += 1
                        else:
                            exercise_counts[ex] = 1
                    
                    # Find most frequent exercise
                    most_common = max(exercise_counts, key=exercise_counts.get)
                    if exercise_counts[most_common] >= 3:  # If the most common exercise appears at least 3 times
                        current_exercise = most_common

        # Display exercise type and count
        cv2.putText(image,
                    f"Exercise: {current_exercise}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.putText(image,
                    f"Count: {exercise_counter}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        #logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        #logger.debug('finished+')

    cv2.destroyAllWindows()