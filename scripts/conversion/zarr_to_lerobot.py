#!/usr/bin/env python
"""
A script to convert robotics data from a single Zarr store into the LeRobot format (v2.1).

This script is designed to process a single Zarr store that contains an entire
dataset, with episode boundaries defined by an `episode_ends` array. It extracts
observations, actions, and state information for each episode and packages them
into a LeRobotDataset, which can then be optionally pushed to the Hugging Face Hub.

Expected Zarr Store Structure:
------------------------------
The script assumes a single Zarr store (e.g., `my_dataset.zarr`) with the following
internal hierarchy. All top-level arrays are expected to be flat, containing
data for all episodes concatenated together. `N` is the total number of steps
across all episodes, and `E` is the total number of episodes.

/
├── action                (Array, shape: (N, 6)): The actions for all steps.
├── observations/
│   └── rgb               (Array, shape: (N, 2, 224, 224, 3)): RGB images for all steps.
├── abs_joint_pos         (Array, shape: (N, 7)): The absolute joint positions for all steps.
├── timestep              (Array, shape: (N,)): The timestamp for each data point.
└── episode_ends          (Array, shape: (E,)): Indices marking the end of each episode.

Usage:
------
To run the script, you can use the following command, pointing to your Zarr store:

    python convert_zarr_to_lerobot.py --data-path /path/to/your/dataset.zarr

To convert and then upload to the Hugging Face Hub:

    python convert_zarr_to_lerobot.py --data-path /path/to/your/dataset.zarr --push-to-hub

Dependencies:
-------------
- lerobot
- tyro
- zarr
- tqdm
"""

import shutil
from pathlib import Path

import zarr
import tqdm
import tyro
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LEROBOT_HOME


def convert_data_to_lerobot(data_path: Path, repo_id: str, *, push_to_hub: bool = False):
    """
    Converts a single Zarr store with episode boundaries to a LeRobotDataset.

    Args:
        data_path: The path to the Zarr store file/directory.
        repo_id: The repository ID for the dataset on the Hugging Face Hub.
        push_to_hub: Whether to push the dataset to the Hub after conversion.
    """
    final_output_path = LEROBOT_HOME / repo_id
    if final_output_path.exists():
        print(f"Removing existing dataset at {final_output_path}")
        shutil.rmtree(final_output_path)

    # Initialize a LeRobotDataset with the desired features.
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        video=True,
        robot_type="panda",
        fps=30,
        features={
            "image": {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"],
            },
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["x", "y", "z", "roll", "pitch", "yaw"],
            },
        },
    )

    print(f"Opening Zarr store at {data_path}")
    try:
        root_zarr = zarr.open(store=str(data_path), mode='r')
    except Exception as e:
        print(f"Error opening Zarr store: {e}")
        return

    if "episode_ends" not in root_zarr:
        print(f"Error: `episode_ends` array not found in {data_path}. Cannot determine episode boundaries.")
        return

    episode_ends = root_zarr["episode_ends"][:]
    num_episodes = len(episode_ends)
    print(f"Found {num_episodes} episodes to convert.")

    # A single, descriptive task for all episodes.
    task_description = "Conduct a liver ultrasound scan"

    # Process each episode based on the episode_ends indices.
    start_idx = 0
    for episode_idx in tqdm.tqdm(range(num_episodes), desc="Converting Episodes"):
        try:
            end_idx = episode_ends[episode_idx]

            # Add each frame from the current episode slice to the dataset buffer.
            for step_idx in range(start_idx, end_idx):
                frame_data = {
                    "image": root_zarr["observations/rgb"][step_idx][0],
                    "wrist_image": root_zarr["observations/rgb"][step_idx][1],
                    "state": root_zarr["abs_joint_pos"][step_idx],
                    "action": root_zarr["action"][step_idx],
                }
                timestamp = root_zarr["timestep"][step_idx]
                dataset.add_frame(frame_data, task=task_description, timestamp=timestamp)

            # Save the buffered frames as a completed episode.
            dataset.save_episode()

            # Update the start index for the next episode.
            start_idx = end_idx

        except Exception as e:
            print(f"Error processing episode {episode_idx}: {e}")
            dataset.clear_episode_buffer()

    print(f"Dataset conversion complete. Saved to {final_output_path}")

    if push_to_hub:
        print(f"Pushing dataset to Hugging Face Hub: {repo_id}")
        dataset.push_to_hub()
        print("Push complete.")


def main(
    data_path: Path = Path("path/to/your/dataset.zarr"),
    repo_id: str = "your-username/your-dataset-name",
    *,
    push_to_hub: bool = False,
):
    """
    Main entry point for the conversion script.

    Args:
        data_path: The path to the single Zarr store for the dataset.
        repo_id: The desired Hugging Face Hub repository ID (e.g., 'username/dataset-name').
        push_to_hub: If True, uploads the dataset to the Hub after conversion.
    """
    if not data_path.exists():
        print(f"Error: The provided Zarr store does not exist: {data_path}")
        print("Please provide a valid path to your Zarr store.")
        return

    if repo_id == "your-username/your-dataset-name":
        print("Warning: Using the default repo_id. Please specify your own with --repo-id.")

    convert_data_to_lerobot(data_path, repo_id, push_to_hub=push_to_hub)


if __name__ == "__main__":
    tyro.cli(main)
