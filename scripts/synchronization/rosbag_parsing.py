#!/usr/bin/env python

"""
ROS Bag Parser for Extracting Time-Series Data from Specific Topics.

This script processes a ROS1 `.bag` file and extracts time-aligned data from selected topics.
It is useful for preparing datasets for machine learning or analysis by converting ROS messages
(e.g., images, transforms) into a structured dictionary format and saving it as a `.pkl` file.

Core Features:
--------------
- Extracts messages and timestamps from specified ROS topics.
- Organizes data in a dictionary with the format:
    - "t_<topic_name>": list of timestamps in seconds
    - "y_<topic_name>": list of associated float-encoded message data
- Saves the extracted data to disk as a Python pickle file.

Placeholders:
-------------
- `image_to_float`: Needs implementation for extracting visual features from image data.
- `tf_to_float`: Needs implementation for converting transformation messages to floats.

Dependencies:
-------------
- ROS (rospy, rosbag)
- OpenCV + cv_bridge
- NumPy
- Pickle

"""


import rosbag
import pickle
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np


def image_to_float(bridge, msg):
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    # --- PLACEHOLDER: extract the pixel coordinate of a visual feature point
    # e.g., cv_image_proc = detect marker or keypoint
    return cv_image_proc # float


def tf_to_float(msg):
    # --- PLACEHOLDER: Extract a fixed metric from msg,
    # e.g., tf_data_proc = norm of end-effector’s translation
    return tf_data_proc # float


def parse_bag(bag_file_path, topics_of_interest, output_pkl_path):
    '''
    Parses a ROS1 bag file and extracts data from selected topics into a dictionary,
    which is then saved as a pickle file. Each topic will produce:
        - t_<topic>: list of timestamps (in seconds)
        - y_<topic>: list of message data (converted where possible)
    Parameters:
    -----------
        bag_file_path (str): Path to the input .bag file.
        topics_of_interest (list of str): List of ROS topics to extract.
        output_pkl_path (str): Path to save the resulting .pkl file.
    '''

    data_dict = {}
    bridge = CvBridge()
    bag = rosbag.Bag(bag_file_path)

    # Iterate over messages in the specified topics
    for topic, msg, t in bag.read_messages(topics=topics_of_interest):
        key_base = topic.strip("/").replace("/", "_")
        t_key = f"t_{key_base}"
        y_key = f"y_{key_base}"
        if t_key not in data_dict:
            data_dict[t_key] = []
            data_dict[y_key] = []

        # Append timestamp (converted to seconds)
        data_dict[t_key].append(t.to_sec())

        # Handle message parsing depending on type/topic
        if isinstance(msg, Image):
            data = image_to_float(bridge, msg)
        elif topic == "/tf":
            data = tf_to_float(msg)
        else:
            data = msg # and many other data streams

        # Append message data
        data_dict[y_key].append(data)

    bag.close()

    # Save to pickle
    with open(output_pkl_path, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"[INFO] Bag parsed and saved to: {output_pkl_path}")