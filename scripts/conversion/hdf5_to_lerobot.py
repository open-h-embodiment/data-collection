#!/usr/bin/env python
"""
A script to convert robotics data from HDF5 files into the LeRobot format (v2.1)
with an efficient MP4 video backend.

This script processes a directory of HDF5 files, where each file represents a
single episode. It extracts observations, actions, and state information, and
packages them into a LeRobotDataset with visual data stored as compressed MP4
videos, then optionally pushes the result to the Hugging Face Hub.

Expected HDF5 File Structure:
------------------------------
The script assumes a directory with zero-indexed HDF5 files (e.g., `data_0.hdf5`).
Each file should contain the following structure:

/data/demo_0/
    ├── action                (Dataset): Actions taken at each step.
    ├── observations/
    │   └── rgb               (Dataset): RGB image observations.
    ├── abs_joint_pos         (Dataset): Absolute joint positions.
    └── timestep              (Dataset): Timestamps for each data point.

Usage:
------
    python convert_data_to_lerobot_video.py --data-dir /path/to/your/hdf5/files --repo-id your-username/your-dataset-name

To also push to the Hub:
    python convert_data_to_lerobot_video.py --data-dir /path/to/your/hdf5/files --repo-id your-username/your-dataset-name --push-to-hub
"""

import glob
import os
import shutil
from pathlib import Path

import h5py
import tqdm
import tyro

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.constants import HF_LEROBOT_HOME


def convert_data_to_lerobot(data_dir: Path, repo_id: str, *, push_to_hub: bool = False):
    """
    Converts a directory of HDF5 files to a LeRobotDataset with a video backend.

    Args:
        data_dir: The path to the directory containing the HDF5 files.
        repo_id: The repository ID for the dataset on the Hugging Face Hub.
        push_to_hub: Whether to push the dataset to the Hub after conversion.
    """
    final_output_path = os.path.join(HF_LEROBOT_HOME, repo_id)
    if final_output_path.exists():
        print(f"Removing existing dataset at {final_output_path}")
        shutil.rmtree(final_output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        use_videos=True,
        robot_type="panda",
        fps=30,
        features={
            "observation.image": {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.wrist_image": {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
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
        image_writer_processes=16,
        image_writer_threads=20,
        tolerance_s=0.1,
    )

    hdf5_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))

    if not hdf5_files:
        print(f"No HDF5 files found in {data_dir}. Exiting.")
        return

    print(f"Found {len(hdf5_files)} episodes to convert.")

    task_description = "Conduct a liver ultrasound scan"

    for hdf5_path in tqdm.tqdm(hdf5_files, desc="Converting Episodes"):
        try:
            with h5py.File(hdf5_path, "r") as f:
                root_name = "data/demo_0"
                if root_name not in f:
                    print(f"Warning: Skipping {hdf5_path} because '{root_name}' group was not found.")
                    continue

                num_steps = len(f[f"{root_name}/action"])

                # Add each frame from the episode to the internal buffer.
                for step in range(num_steps):
                    frame_data = {
                        "observation.image": f[f"{root_name}/observations/rgb"][step],
                        "observation.wrist_image": f[f"{root_name}/observations/rgb"][step],
                        "observation.state": f[f"{root_name}/abs_joint_pos"][step],
                        "action": f[f"{root_name}/action"][step],
                    }
                    timestamp = f[f"{root_name}/timestep"][step]
                    dataset.add_frame(frame_data, task=task_description, timestamp=timestamp)

            # After processing all frames for an HDF5 file, save the buffered
            # data as a completed episode. This will trigger the video encoding
            # for the 'image' and 'wrist_image' frames collected.
            dataset.save_episode()

        except Exception as e:
            print(f"Error processing {hdf5_path}: {e}")
            # It's good practice to clear the buffer on error to prevent
            # a failed episode from contaminating the next one.
            dataset.clear_episode_buffer()

    print(f"Dataset conversion complete. Saved to {final_output_path}")

    if push_to_hub:
        print(f"Pushing dataset to Hugging Face Hub: {repo_id}")
        dataset.push_to_hub()
        print("Push complete.")


def main(
    data_dir: Path = Path("path/to/your/data"),
    repo_id: str = "your-username/your-dataset-name",
    *,
    push_to_hub: bool = False,
):
    """
    Main entry point for the conversion script.

    Args:
        data_dir: The directory containing HDF5 episode files.
        repo_id: The desired Hugging Face Hub repository ID.
        push_to_hub: If True, uploads the dataset to the Hub.
    """
    if not data_dir.is_dir():
        print(f"Error: The provided data directory does not exist: {data_dir}")
        return

    if repo_id == "your-username/your-dataset-name":
        print("Warning: Using the default repo_id. Please specify your own with --repo-id.")

    convert_data_to_lerobot(data_dir, repo_id, push_to_hub=push_to_hub)


if __name__ == "__main__":
    tyro.cli(main)