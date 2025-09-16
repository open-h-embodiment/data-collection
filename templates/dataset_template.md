<!--
Open-H Embodiment Dataset README Template (v1.0)
Please fill out this template and include it in the ./metadata directory of your LeRobot dataset.
This file helps others understand the context and details of your contribution.
-->

# [Dataset Name] - README

---

## 📋 At a Glance

*Provide a one-sentence summary of your dataset.*

**Example:** *Teleoperated demonstrations of a da Vinci robot performing needle passing on a silicone phantom.*

---

## 📖 Dataset Overview

*Briefly describe the purpose and content of this dataset. What key skills or scenarios does it demonstrate?*

**Example:** *This dataset contains 2,500 trajectories of expert surgeons using the dVRK to perform surgical suturing tasks. It includes successful trials, failures, and recovery attempts to provide a robust dataset for training imitation learning policies.*

| | |
| :--- | :--- |
| **Total Trajectories** | `[Number]` |
| **Total Hours** | `[Number]` |
| **Data Type** | `[ ] Clinical` `[ ] Ex-Vivo` `[ ] Table-Top Phantom` `[ ] Digital Simulation` `[ ] Physical Simulation` `[ ] Other (If checked, update "Other")` |
| **License** | CC BY 4.0 |
| **Version** | `[e.g., 1.0]` |

---

## 🎯 Tasks & Domain

### Domain

*Select the primary domain for this dataset.*

- [ ] **Surgical Robotics**
- [ ] **Ultrasound Robotics**
- [ ] **Other Healthcare Robotics** (Please specify: `[Your Domain]`)

### Demonstrated Skills

*List the primary skills or procedures demonstrated in this dataset.*

***Example:***
- Needle-passing
- Suture-tying
- ...

---

## 🔬 Data Collection Details

### Collection Method

*How was the data collected?*

- [ ] **Human Teleoperation**
- [ ] **Programmatic/State-Machine**
- [ ] **AI Policy / Autonomous**
- [ ] **Other** (Please specify: `[Your Method]`)

### Operator Details

| | Description |
| :--- | :--- |
| **Operator Count** | `[Number of unique people who collected data]` |
| **Operator Skill Level** | `[ ] Expert (e.g., Surgeon, Sonographer)` <br> `[ ] Intermediate (e.g., Trained Researcher)` <br> `[ ] Novice (e.g., ML Researcher with minimal experience)` <br> `[ ] N/A` |
| **Collection Period** | From `[YYYY-MM-DD]` to `[YYYY-MM-DD]` |

### Recovery Demonstrations

*Does this dataset include examples of recovering from failure?*

- [ ] **Yes**
- [ ] **No**

**If yes, please briefly describe the recovery process:**

*Example: For 250 demonstrations, demonstrations are initialized from a failed needle grasp position, the operator re-orients the robotic grippers and attempts to grasp the needle again from a different angle.*

---

## 💡 Diversity Dimensions

*Check all dimensions that were intentionally varied during data collection.*

- [ ] **Camera Position / Angle**
- [ ] **Lighting Conditions**
- [ ] **Target Object** (e.g., different phantom models, suture types)
- [ ] **Spatial Layout** (e.g., placing the target suture needle in various locations)
- [ ] **Robot Embodiment** (if multiple robots were used)
- [ ] **Task Execution** (e.g., different techniques for the same task)
- [ ] **Background / Scene**
- [ ] **Other** (Please specify: `[Your Dimension]`)

*If you checked any of the above please briefly elaborate below.*

**Example:** We adjusted the room camera perspective every 100 demonstrations. The camera angle was varied by panning up and down by +/- 10 degrees, as well as manually adjusting the height of the camera mount by +/- 2 cm. Additionally, we varied the needle used by swapping out various curvatures, including 1/4, 3/8, 1/2, and 5/8.

---

## 🛠️ Equipment & Setup

### Robotic Platform(s)

*List the primary robot(s) used.*

- **Robot 1:** `[e.g., dVRK (da Vinci Research Kit)]`
- **Robot 2:** `[If applicable]`

### Sensors & Cameras

*List the sensors and cameras used. Specify model names where possible. (Add and remove rows as needed)*

| Type | Model/Details |
| :--- | :--- |
| **Primary Camera** | `[e.g., Endoscopic Camera, 1920x1080 @ 30fps]` |
| **Room/3rd Person Camera** | `[e.g., Logitech C920, 1920x1080 @ 30fps]` |
| **Force/Torque Sensor** | `[e.g., ATI Nano25]` |
| **Medical Imager** | `[e.g., GE Voluson E10 Ultrasound, B-Mode]` |
| **Other** | `[Specify]` |

---

## 🎯 Action & State Space Representation

*Describe how actions and robot states are represented in your dataset. This is crucial for understanding data compatibility and enabling effective policy learning.*

### Action Space Representation

**Primary Action Representation:**
- [ ] **Absolute Cartesian** (position/orientation relative to robot base)
- [ ] **Relative Cartesian** (delta position/orientation from current pose)
- [ ] **Joint Space** (direct joint angle commands)
- [ ] **Other** (Please specify: `[Your Representation]`)

**Orientation Representation:**
- [ ] **Quaternions** (x, y, z, w)
- [ ] **Euler Angles** (roll, pitch, yaw)
- [ ] **Axis-Angle** (rotation vector)
- [ ] **Rotation Matrix** (3x3 matrix)
- [ ] **Other** (Please specify: `[Your Representation]`)

**Reference Frame:**
- [ ] **Robot Base Frame**
- [ ] **Tool/End-Effector Frame**
- [ ] **World/Global Frame**
- [ ] **Camera Frame**
- [ ] **Other** (Please specify: `[Your Frame]`)

**Action Dimensions:**
*List the action space dimensions and their meanings.*

**Example:**
```
action: [x, y, z, qx, qy, qz, qw, gripper]
- x, y, z: Absolute position in robot base frame (meters)
- qx, qy, qz, qw: Absolute orientation as quaternion
- gripper: Gripper opening angle (radians)
```

### State Space Representation

**State Information Included:**
- [ ] **Joint Positions** (all articulated joints)
- [ ] **Joint Velocities**
- [ ] **End-Effector Pose** (Cartesian position/orientation)
- [ ] **Force/Torque Readings**
- [ ] **Gripper State** (position, force, etc.)
- [ ] **Other** (Please specify: `[Your State Info]`)

**State Dimensions:**
*List the state space dimensions and their meanings.*

**Example:**
```
observation.state: [j1, j2, j3, j4, j5, j6, j7, gripper_pos]
- j1-j7: Absolute joint positions for 7-DOF arm (radians)
- gripper_pos: Current gripper opening (meters)
```

### 📋 Recommended Additional Representations

*Even if not your primary action/state representation, we strongly encourage including these standardized formats for maximum compatibility:*

**Recommended Action Fields:**
- **`action.cartesian_absolute`**: Absolute Cartesian pose with absolute quaternions
  ```
  [x, y, z, qx, qy, qz, qw, gripper_angle]
  ```

**Recommended State Fields:**
- **`observation.state.joint_positions`**: Absolute positions for all articulated joints
  ```
  [joint_1, joint_2, ..., joint_n]
  ```


---

## ⏱️ Data Synchronization Approach

*Describe how you achieved proper data synchronization across different sensors, cameras, and robotic systems during data collection. This is crucial for ensuring temporal alignment of all modalities in your dataset.*

**Example:** *We collect joint kinematics from our Franka Research 3 and RGB-D frames from Intel RealSense D435 cameras, all running in ROS 2 Galactic on the same workstation clocked with ROS Time. Both drivers stamp their outgoing messages’ header.stamp fields with the shared system clock, and we record /joint_states, /camera/*/image_raw, and /camera/*/camera_info in a single rosbag2 session. During export to LeRobot, each data point’s ROS header.stamp is written verbatim into the timestamp attribute. Offline checks show inter-sensor skew stays below ±2 ms across a 2-minute capture.*

---

## 👥 Attribution & Contact

*Please provide attribution for the dataset creators and a point of contact.*

| | |
| :--- | :--- |
| **Dataset Lead** | `[Name1, Name2, ...]` |
| **Institution** | `[Your Institution]` |
| **Contact Email** | `[email1@example.com, email2@example.com, ...]` |
| **Citation (BibTeX)** | <pre><code>@misc{[your_dataset_name_2025],<br>  author = {[Your Name(s)]},<br>  title = {[Your Dataset Title]},<br>  year = {2025},<br>  publisher = {Open-H-Embodiment},<br>  note = {https://hrpp.research.virginia.edu/teams/irb-sbs/researcher-guide-irb-sbs/identifiers}<br>}</code></pre> |
