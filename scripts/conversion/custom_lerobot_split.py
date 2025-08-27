#!/usr/bin/env python

import glob
import os
from pathlib import Path
import h5py
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_info


def add_episodes_from_dir(dataset, data_dir: Path, num_expected: int, task_description: str):
    """Helper: read episodes from HDF5 directory and add to dataset."""
    hdf5_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))

    print(f"Found {len(hdf5_files)} episodes in {data_dir}")
    assert len(hdf5_files) == num_expected, "Mismatch with expected episode count"

    for hdf5_path in tqdm.tqdm(hdf5_files, desc=f"Loading {data_dir.name}"):
        with h5py.File(hdf5_path, "r") as f:
            root_name = "data/demo_0"
            num_steps = len(f[f"{root_name}/action"])

            for step in range(num_steps):
                frame_data = {
                    "image": f[f"{root_name}/observations/rgb"][step],
                    "wrist_image": f[f"{root_name}/observations/rgb"][step],
                    "state": f[f"{root_name}/abs_joint_pos"][step],
                    "action": f[f"{root_name}/action"][step],
                }
                timestamp = f[f"{root_name}/timestep"][step]
                dataset.add_frame(frame_data, task=task_description, timestamp=timestamp)

        dataset.save_episode()  # finalize this episode


def main():
    # -----------------------
    # 1. Create fresh dataset
    # -----------------------
    dataset = LeRobotDataset.create(
        repo_id="my_robot_dataset",
        use_videos=True,
        fps=30,
        features={
            "image": {"dtype": "video", "shape": (224, 224, 3), "names": ["h", "w", "c"]},
            "wrist_image": {"dtype": "video", "shape": (224, 224, 3), "names": ["h", "w", "c"]},
            "state": {"dtype": "float32", "shape": (7,), "names": [f"joint_{i}" for i in range(1, 8)]},
            "action": {"dtype": "float32", "shape": (6,), "names": ["x", "y", "z", "roll", "pitch", "yaw"]},
        },
        robot_type="panda",
        image_writer_processes=16,
        image_writer_threads=20,
        tolerance_s=0.1,
    )

    # -------------------------------
    # 2. Load main training episodes
    # -------------------------------
    add_episodes_from_dir(dataset, Path("data/main"), num_expected=125, task_description="normal task")

    # -------------------------------
    # 3. Load recovery episodes
    # -------------------------------
    add_episodes_from_dir(dataset, Path("data/main_recovery"), num_expected=15, task_description="recovery")

    # -------------------------------
    # 4. Load failure episodes
    # -------------------------------
    add_episodes_from_dir(dataset, Path("data/failure"), num_expected=10, task_description="failure")

    # --------------------------------------
    # 5. Write custom splits into info.json
    # --------------------------------------
    dataset.meta.info["splits"] = {
        "train": "0:85",
        "val": "85:100",
        "test": "100:125",
        "recovery": "125:140",  # recovery episodes
        "failure": "140:150",   # failure episodes
    }
    write_info(dataset.meta.info, dataset.root)

    print("Custom split configuration saved!")


if __name__ == "__main__":
    main()
