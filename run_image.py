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

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

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

# Find closest matching sample in the landmarks dataset
def find_similar_pose(features, landmarks_df, labels_df):
    if features is None or len(landmarks_df) == 0:
        return None
    
    # In a real implementation, you would use a more sophisticated similarity metric
    # and proper feature engineering based on the landmarks dataset
    
    # For demonstration purposes, we're returning a simplified result
    exercise_type = classify_exercise(features, landmarks_df, labels_df)
    
    # Find a sample video ID for this exercise type
    matching_video_ids = labels_df[labels_df['class'] == exercise_type]['vid_id'].tolist()
    if matching_video_ids:
        return {
            'exercise': exercise_type,
            'confidence': 0.85,  # Placeholder confidence
            'similar_video_id': matching_video_ids[0]
        }
    return None

def process_image(image_path, tfpose_estimator, labels_df, landmarks_df, output_path=None):
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return
    
    logger.info(f'Image loaded: {image_path}, shape={image.shape}')
    
    # Run pose estimation
    humans = tfpose_estimator.inference(image, resize_to_default=True, upsample_size=4.0)
    
    # Draw skeleton on image
    image_with_skeleton = TfPoseEstimator.draw_humans(image.copy(), humans, imgcopy=False)
    
    # Extract features and classify exercise
    current_exercise = "Unknown"
    similar_pose_info = None
    
    if humans:
        features = extract_pose_features(humans)
        if features:
            current_exercise = classify_exercise(features, landmarks_df, labels_df)
            similar_pose_info = find_similar_pose(features, landmarks_df, labels_df)
    
    # Display exercise type
    cv2.putText(image_with_skeleton,
                f"Exercise: {current_exercise}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
    
    # Display similar pose info if found
    if similar_pose_info:
        cv2.putText(image_with_skeleton,
                    f"Confidence: {similar_pose_info['confidence']:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.putText(image_with_skeleton,
                    f"Similar video ID: {similar_pose_info['similar_video_id']}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
    
    # Save or display the result
    if output_path:
        cv2.imwrite(output_path, image_with_skeleton)
        logger.info(f'Result saved to: {output_path}')
    else:
        cv2.imshow('Result', image_with_skeleton)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Return the classification results
    return {
        'exercise_type': current_exercise,
        'similar_pose': similar_pose_info
    }

def process_webcam(cam_source, tfpose_estimator, labels_df, landmarks_df):
    cam = cv2.VideoCapture(cam_source)
    ret_val, image = cam.read()
    if not ret_val:
        logger.error("Failed to open webcam")
        return
    
    logger.info(f'Webcam opened, image shape={image.shape}')
    
    fps_time = 0
    while True:
        ret_val, image = cam.read()
        if not ret_val:
            break
            
        # Process frame
        humans = tfpose_estimator.inference(image, resize_to_default=True, upsample_size=4.0)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        
        # Extract features and classify exercise
        current_exercise = "Unknown"
        if humans:
            features = extract_pose_features(humans)
            if features:
                current_exercise = classify_exercise(features, landmarks_df, labels_df)
        
        # Display exercise type
        cv2.putText(image,
                    f"Exercise: {current_exercise}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        
        # Display FPS
        cv2.putText(image,
                    f"FPS: {1.0 / (time.time() - fps_time):.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        fps_time = time.time()
        
        # Show result
        cv2.imshow('Result', image)
        if cv2.waitKey(1) == 27:  # ESC key
            break
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation for images and webcam')
    parser.add_argument('--camera', type=str, default=None,
                        help='Camera index or image file path')
    parser.add_argument('--image', type=str, default=None,
                        help='Image file path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image file path (only used with --image)')
    parser.add_argument('--model', type=str, default='mobilenet_thin', 
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--labels', type=str, default='Dataset/labels.csv',
                        help='path to the labels.csv file')
    parser.add_argument('--landmarks', type=str, default='Dataset/landmarks.csv',
                        help='path to the landmarks.csv file')
    
    args = parser.parse_args()
    
    # Validate input
    if args.camera is None and args.image is None:
        parser.error('Either --camera or --image must be specified')
    
    # Load datasets
    labels_df, landmarks_df, exercise_classes = load_datasets(args.labels, args.landmarks)
    if labels_df is None or landmarks_df is None:
        logger.error("Failed to load datasets. Exiting.")
        exit(1)
    
    # Initialize TF Pose Estimator
    logger.info(f'Initializing pose estimator with model: {args.model}')
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    
    # Process input
    if args.image is not None:
        # Process a single image
        result = process_image(args.image, e, labels_df, landmarks_df, args.output)
        logger.info(f"Image processing result: {result}")
    elif args.camera is not None:
        # Process webcam or video
        try:
            # Try to convert to integer (webcam index)
            cam_source = int(args.camera)
        except ValueError:
            # Use as file path
            cam_source = args.camera
        
        process_webcam(cam_source, e, labels_df, landmarks_df)