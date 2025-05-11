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
def load_datasets(labels_path, landmarks_path, angles_path):
    try:
        # Load labels dataset
        labels_df = pd.read_csv(labels_path)
        logger.info(f'Loaded labels dataset with {len(labels_df)} entries')
        
        # Load landmarks dataset
        landmarks_df = pd.read_csv(landmarks_path)
        logger.info(f'Loaded landmarks dataset with {len(landmarks_df)} entries')
        
        # Load angles dataset
        angles_df = pd.read_csv(angles_path)
        logger.info(f'Loaded angles dataset with {len(angles_df)} entries')
        
        # Extract unique exercise classes
        exercise_classes = labels_df['class'].unique() if 'class' in labels_df.columns else []
        logger.info(f'Found exercise classes: {exercise_classes}')
        
        return labels_df, landmarks_df, angles_df, exercise_classes
    except Exception as e:
        logger.error(f'Error loading datasets: {e}')
        return None, None, None, None

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

def extract_body_angles(humans):
    """Extract key body angles from pose estimation results"""
    if not humans or len(humans) == 0:
        return None
    
    # Use the first detected person
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
    for i, body_part in body_parts.items():
        keypoints[i] = (body_part.x, body_part.y)
    
    # Calculate mid-points for some joint pairs
    if 5 in keypoints and 2 in keypoints:  # Shoulders
        keypoints['mid_shoulder'] = ((keypoints[5][0] + keypoints[2][0])/2,
                                    (keypoints[5][1] + keypoints[2][1])/2)
    
    if 11 in keypoints and 8 in keypoints:  # Hips
        keypoints['mid_hip'] = ((keypoints[11][0] + keypoints[8][0])/2,
                              (keypoints[11][1] + keypoints[8][1])/2)
    
    # Calculate angles of interest
    
    # Right arm angle (shoulder to elbow to wrist)
    if 2 in keypoints and 3 in keypoints and 4 in keypoints:
        angles['right_arm'] = calculate_angle(keypoints[2], keypoints[3], keypoints[4])
    
    # Left arm angle (shoulder to elbow to wrist)
    if 5 in keypoints and 6 in keypoints and 7 in keypoints:
        angles['left_arm'] = calculate_angle(keypoints[5], keypoints[6], keypoints[7])
    
    # Right leg angle (hip to knee to ankle)
    if 8 in keypoints and 9 in keypoints and 10 in keypoints:
        angles['right_leg'] = calculate_angle(keypoints[8], keypoints[9], keypoints[10])
    
    # Left leg angle (hip to knee to ankle)
    if 11 in keypoints and 12 in keypoints and 13 in keypoints:
        angles['left_leg'] = calculate_angle(keypoints[11], keypoints[12], keypoints[13])
    
    # Right hip angle (shoulder to hip to knee)
    if 'mid_shoulder' in keypoints and 8 in keypoints and 9 in keypoints:
        angles['right_hip'] = calculate_angle(keypoints['mid_shoulder'], keypoints[8], keypoints[9])
    
    # Left hip angle (shoulder to hip to knee)
    if 'mid_shoulder' in keypoints and 11 in keypoints and 12 in keypoints:
        angles['left_hip'] = calculate_angle(keypoints['mid_shoulder'], keypoints[11], keypoints[12])
    
    # Torso angle (neck to hip)
    if 1 in keypoints and 'mid_hip' in keypoints:
        # Calculate angle relative to vertical (90 degrees is standing straight)
        # This is a simplified approach
        dx = keypoints[1][0] - keypoints['mid_hip'][0]
        dy = keypoints[1][1] - keypoints['mid_hip'][1]
        angles['torso'] = np.degrees(np.arctan2(dx, -dy))  # Negative dy because y-axis is down in images
        # Convert to 0-180 range
        if angles['torso'] < 0:
            angles['torso'] += 180
    
    # Calculate angles for specific exercise recognition (matching angles.csv)
    if 3 in keypoints and 2 in keypoints and 8 in keypoints:
        angles['right_elbow_right_shoulder_right_hip'] = calculate_angle(keypoints[3], keypoints[2], keypoints[8])
    
    if 6 in keypoints and 5 in keypoints and 11 in keypoints:
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
    
    # Return all calculated angles and the keypoints for visualization
    return {'angles': angles, 'keypoints': keypoints}

def classify_exercise_by_angles(angles):
    """
    Classify the exercise based on joint angles using simple rule-based approach
    """
    if not angles:
        return "Unknown"
    
    # Get average arm and leg angles if available
    arm_angles = []
    if 'right_arm' in angles:
        arm_angles.append(angles['right_arm'])
    if 'left_arm' in angles:
        arm_angles.append(angles['left_arm'])
    
    leg_angles = []
    if 'right_leg' in angles:
        leg_angles.append(angles['right_leg'])
    if 'left_leg' in angles:
        leg_angles.append(angles['left_leg'])
    
    # Need at least one arm and one leg angle for classification
    if not arm_angles or not leg_angles:
        return "Unknown"
    
    avg_arm_angle = sum(arm_angles) / len(arm_angles)
    avg_leg_angle = sum(leg_angles) / len(leg_angles)
    
    # Check for squat
    if 'right_hip_right_knee_right_ankle' in angles and 'left_hip_left_knee_left_ankle' in angles:
        knee_angle = (angles['right_hip_right_knee_right_ankle'] + angles['left_hip_left_knee_left_ankle']) / 2
        if 70 < knee_angle < 130:
            return "squat"
    
    # Check for push-up
    if 'torso' in angles and 'right_arm' in angles and 'left_arm' in angles:
        # In push-up, arms are bent and torso is fairly horizontal
        if 60 < avg_arm_angle < 120 and 45 < angles['torso'] < 135:
            return "push_up"
    
    # # Check for jumping jack
    # if 'right_arm' in angles and 'left_arm' in angles:
    #     if avg_arm_angle > 140:  # Arms raised up/out
    #         return "jumping_jack"
    
    # # Check for situp
    # if 'torso' in angles and 30 < angles['torso'] < 90:
    #     return "situp"
    
    # # Check for pull-up
    # if avg_arm_angle < 90 and 'torso' in angles and angles['torso'] > 150:
    #     return "pull_up"
    
    return "Unknown"

def detect_exercise_state(angles, current_exercise):
    """
    Determine the current state (up/down) of an exercise based on angles
    """
    if not angles or current_exercise == "Unknown":
        return "unknown"
    
    if current_exercise == "squat":
        if 'right_hip_right_knee_right_ankle' in angles and 'left_hip_left_knee_left_ankle' in angles:
            knee_angle = (angles['right_hip_right_knee_right_ankle'] + angles['left_hip_left_knee_left_ankle']) / 2
            if knee_angle < 100:
                return "down"
            elif knee_angle > 160:
                return "up"
    
    elif current_exercise == "push_up":
        if 'right_arm' in angles and 'left_arm' in angles:
            arm_angle = (angles['right_arm'] + angles['left_arm']) / 2
            
            if arm_angle < 90:
                return "down"
            elif arm_angle > 150:
                return "up"
            else:
                return "transitioning"

    
    # elif current_exercise == "situp":
    #     if 'torso' in angles:
    #         if angles['torso'] > 60:
    #             return "up"
    #         elif angles['torso'] < 30:
    #             return "down"
    
    # elif current_exercise == "jumping_jack":
    #     if 'right_arm' in angles and 'left_arm' in angles:
    #         arm_angle = (angles['right_arm'] + angles['left_arm']) / 2
    #         if arm_angle > 140:
    #             return "up"
    #         elif arm_angle < 80:
    #             return "down"
    
    # elif current_exercise == "pull_up":
    #     if 'right_arm' in angles and 'left_arm' in angles:
    #         arm_angle = (angles['right_arm'] + angles['left_arm']) / 2
    #         if arm_angle < 90:
    #             return "up"
    #         elif arm_angle > 150:
    #             return "down"
    
    return "transitioning"  # In between up and down

def count_reps(exercise, new_state, prev_state, rep_counts):
    """
    Count repetitions based on state transitions
    """
    if exercise == "Unknown" or new_state == "unknown" or prev_state == "unknown":
        return rep_counts
    
    if exercise not in rep_counts:
        rep_counts[exercise] = 0
    
    # Count a rep when transitioning from down to up for most exercises
    if prev_state == "down" and new_state == "up":
        rep_counts[exercise] += 1
    
    return rep_counts

def debug_angles(image, angles, y_start=130):
    """Display angle values on the image for debugging"""
    if not angles:
        return image
    
    y_pos = y_start
    for angle_name, angle_value in angles.items():
        # Limit to showing 10 angles to avoid cluttering the screen
        if y_pos > y_start + 200:
            break
            
        cv2.putText(image,
                f"{angle_name}: {angle_value:.1f}°",
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255,105,180), 1)
        y_pos += 20
    
    return image

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
    parser.add_argument('--labels', type=str, default='Dataset/labels.csv',
                        help='path to the labels.csv file')
    parser.add_argument('--landmarks', type=str, default='Dataset/landmarks.csv',
                        help='path to the landmarks.csv file')
    parser.add_argument('--angles', type=str, default='Dataset/angles.csv',
                        help='path to the angles.csv file')
    parser.add_argument('--debug', action='store_true',
                        help='show debug information including all angles')
    args = parser.parse_args()

    # Load datasets
    labels_df, landmarks_df, angles_df, exercise_classes = load_datasets(args.labels, args.landmarks, args.angles)

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
    
    # For rep counting
    rep_counts = {}
    last_state = "unknown"
    state_history = deque(maxlen=5)  # To stabilize state detection

    while True:
        ret_val, image = cam.read()
        if not ret_val:
            logger.error("Failed to read from camera")
            break

        #logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        #logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        # Extract angles and classify exercise if humans detected
        if humans:
            body_data = extract_body_angles(humans)
            
            if body_data and 'angles' in body_data:
                # Get angles
                angles = body_data['angles']
                
                # Classify exercise
                detected_exercise = classify_exercise_by_angles(angles)
                pose_history.append(detected_exercise)
                
                # Stabilize exercise detection
                if len(pose_history) >= 5:
                    exercise_counts = {}
                    for ex in pose_history:
                        exercise_counts[ex] = exercise_counts.get(ex, 0) + 1
                    
                    most_common = max(exercise_counts, key=exercise_counts.get)
                    if exercise_counts[most_common] >= 3:  # At least 3 out of 5 frames agree
                        current_exercise = most_common
                
                # Detect exercise state
                current_state = detect_exercise_state(angles, current_exercise)
                state_history.append(current_state)
                
                # Stabilize state detection
                if len(state_history) >= 3:
                    state_counts = {}
                    for state in state_history:
                        state_counts[state] = state_counts.get(state, 0) + 1
                    
                    stable_state = max(state_counts, key=state_counts.get)
                    if state_counts[stable_state] >= 2:  # At least 2 out of 3 frames agree
                        # Only count rep if the state is stable and changed from the last state
                        if stable_state != last_state:
                            rep_counts = count_reps(current_exercise, stable_state, last_state, rep_counts)
                            last_state = stable_state
                
                # Debug: display angles
                if args.debug:
                    image = debug_angles(image, angles)
        
        # Display exercise type and count
        cv2.putText(image,
                    f"Exercise: {current_exercise}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,105,180), 2)
        
        # Display the rep count for the current exercise
        # count_display = rep_counts.get(current_exercise, 0) if current_exercise != "Unknown" else 0
        # cv2.putText(image,
        #             f"Count: {count_display}",
        #             (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 255, 0), 2)
        
        # Display the state for the current exercise
        cv2.putText(image,
                    f"State: {last_state}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,105,180), 2)

        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,105,180), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:  # ESC key
            break

    cv2.destroyAllWindows()