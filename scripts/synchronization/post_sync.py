#!/usr/bin/env python

"""
Script for extracting and synchronizing messages from a ROS bag file.

This tool is useful for robot learning applications where sensor data (e.g., images, poses) from different topics
needs to be temporally aligned. It provides a way to:
1. Load messages from specified ROS topics in a bag file.
2. Apply optional time offsets to compensate for sensor delays.
3. Synchronize messages across topics based on a configurable time slop (tolerance).
4. Export the synchronized data for further processing or dataset generation (e.g., in HDF5 format).

Key Components:
- `SyncedRosbagExtractor`: Main class that handles message loading, time offset correction, and temporal synchronization.
- Uses `CvBridge` for converting ROS image messages to OpenCV format.
- Demonstration at the bottom shows how to synchronize pose and image data, and outlines placeholders for saving the result.

Typical Use Case:
- You recorded a demonstration or trajectory using a robot and external sensors (e.g., a camera).
- These sensors publish to ROS topics at different frequencies and potentially with slight timestamp mismatches.
- You want to extract time-synchronized data tuples (e.g., {image, pose}) to train a model or analyze behavior.

Usage Notes:
- Requires topic types and (optionally) per-topic time offsets to be specified.
- Synchronization is tolerant to time differences within `slop` seconds (default: 0.03s).
- Output saving to `.hdf5` is outlined but left as placeholders for user customization.

Dependencies:
- `rospy`, `rosbag`, `sensor_msgs`, `geometry_msgs`, `cv_bridge`, `h5py`

"""

import rospy
import rosbag
from cv_bridge import CvBridge

class SyncedRosbagExtractor:
    def __init__(self, topics=None, slop=0.03, time_offsets=None):
        """
        Initialize the rosbag extractor.
        :param topics: List of tuples (topic_name, message_type).
        :param slop: Maximum allowed time difference between messages (in seconds).
        :param time_offsets: Dictionary of time offsets for each topic (in seconds).
        """
        self.topics = [topic for topic, _ in topics]
        self.slop = rospy.Duration(slop)
        self.time_offsets = time_offsets if time_offsets else {}
        self.bridge = CvBridge()


    def load_messages(self, bag_path):
        """
        Load messages from the rosbag and apply time offsets.
        :param bag_path: Path to the rosbag file.
        """
        bag = rosbag.Bag(bag_path, 'r')
        self.msgs_dict = {topic: [] for topic in self.topics}
        for topic, msg, t in bag.read_messages(topics=self.topics):
            # Apply time offset (in seconds), default to 0.0
            offset = self.time_offsets.get(topic, 0.0)
            adjusted_time = t + rospy.Duration(offset)
            self.msgs_dict[topic].append((adjusted_time, msg))
        bag.close()
        # Sort messages per topic by adjusted timestamp
        for topic in self.topics:
            self.msgs_dict[topic].sort(key=lambda x: x[0])


    def synchronize_messages(self):
        """
        Synchronize messages from different topics based on their (adjusted) timestamps.
        :return: Generator yielding tuples of synchronized messages 
                 in the order of self.topics.
        """
        pointers = {topic: 0 for topic in self.topics}


        while all(pointers[topic] < len(self.msgs_dict[topic]) for topic in self.topics):
            current_msgs = {}
            current_times = {}
            for topic in self.topics:
                t, msg = self.msgs_dict[topic][pointers[topic]]
                current_msgs[topic] = msg
                current_times[topic] = t
            t_min = min(current_times.values())
            t_max = max(current_times.values())
            if (t_max - t_min) <= self.slop:
	          # TODO: can be modified to yield the timestamp
                yield tuple(current_msgs[topic] for topic in self.topics)
                for topic in self.topics:
                    pointers[topic] += 1
            else:
                for topic in self.topics:
                    if current_times[topic] == t_min:
                        pointers[topic] += 1
                        break


import h5py
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
if __name__ == '__main__':
    '''
    An example of a rosbag info:
    types:      geometry_msgs/PoseStamped [d3812c3cbc69362b77dc0b19b345f8f5]
                sensor_msgs/Image         [060021388200f6f0f447d0fcd9c64743]
    topics:     /end_effector_pose              929 msgs    : geometry_msgs/PoseStamped
                /front_camera/color/image_raw   278 msgs    : sensor_msgs/Image    
    '''
    # Suppose you measured:
    reference_topic = "/front_camera/color/image_raw"
    time_offsets = {
        reference_topic: 0.0,
        "/end_effector_pose": 0.1,
    }
    topics_list = [
        ("/end_effector_pose", PoseStamped),
        ("/front_camera/color/image_raw", Image),
    ]
    bagpath = 'path/to/the/bagfile.bag'
   
    extractor = SyncedRosbagExtractor(
        topics=topics_list, slop=0.03, time_offsets=time_offsets
    )
    extractor.load_messages(bagpath)


    # stream the synced data and save it into an HDF5 file
    episode_dict = dict()
    for synced_msgs in extractor.synchronize_messages():
        # synced_msgs is a tuple of messages corresponding to self.topics order
        topic = topic[0]
        for topic, msg in zip(topics_list, synced_msgs):
           # PLACEHOLDER init episode_dict
           if topic == "/front_camera/color/image_raw":
                img = extractor.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
		    # PLACEHOLDER for appending the visual data
		    # episode_dict[topic].append(img)
           elif topic == "/end_effector_pose":
                # PLACEHOLDER for appending the pose data
          # episode_dict[topic].append(pose)


    with h5py.File(save_path+'.hdf5', 'w', rdcc_nbytes=1024**2*2) as h5f:
	  # PLACEHOLDER dump the episode_dict into the hdf5 file
	  # ...