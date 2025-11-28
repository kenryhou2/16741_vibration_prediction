# ROS2 Bag Parser for UR Joint States, TCP Pose/Twist, and Mocap Rigid Bodies

This script converts a **ROS2 rosbag2** (e.g., `.mcap`) into structured **NumPy arrays** for
machine learning, state estimation, and plotting. It automatically extracts relevant robot
and mocap signals **within a specific time window**, converts poses to RPY, and optionally
plots the results.

---

## Usage

```bash
python3 bag_parse.py <rosbag_directory> \
    --joint-topic /ur/joint_states \
    --tcp-pose-topic /ur/tcp_pose_current \
    --tcp-twist-topic /ur/tcp_twist_current \
    --rigid-topic /rigid_bodies \
    --rigid-index 1 \
    --cmd-topic /waypoint_publisher/command \
    --status-topic /waypoint_publisher/status \
    --storage-id mcap \
    --start-offset -0.5 \
    --end-offset 2.0 \
    --output output_data.npz \
    --plot
```
---

## ✨ Features

### ✔ Automatic time-window selection  
The script extracts signals only within:

```
[first /waypoint_publisher/command  →  first /waypoint_publisher/status]
```

You may also manually expand/shrink the window using:

- `--start-offset <seconds>`
- `--end-offset <seconds>`

(Offsets can be negative.)

---

## ✔ Extracted Signals

### **1. Joint States** (`/ur/joint_states`)
Saved fields:
- `time_rel`
- `ros_time_ns`
- `positions` (N×6)
- `velocities` (N×6)
- `efforts` (N×6)
- `joint_names`

---

### **2. TCP Pose** (`/ur/tcp_pose_current`)
Saved fields:
- `tcp_time_rel`
- `tcp_ros_time_ns`
- `tcp_position` (N×3)
- `tcp_orientation_xyzw` (N×4)
- `tcp_rpy` (N×3)
- `tcp_frame_ids`

---

### **3. TCP Twist** (`/ur/tcp_twist_current`)
Saved fields:
- `tcp_twist_time_rel`
- `tcp_twist_ros_time_ns`
- `tcp_twist_linear` (N×3)
- `tcp_twist_angular` (N×3)
- `tcp_twist_frame_ids`

---

### **4. Mocap Rigid Body Pose** (`/rigid_bodies`)
Extracts:
```
rigidbodies[rigid_index].pose
```
Saved fields:
- `rb_time_rel`
- `rb_ros_time_ns`
- `rb_position` (N×3)
- `rb_orientation_xyzw` (N×4)
- `rb_rpy` (N×3)
- `rb_frame_ids`
- `rb_name`

If the message type `mocap4r2_msgs` is unavailable, this extraction is skipped automatically.

---

## Output

All extracted data is stored in a **single `.npz` archive**:

```
<bag_dir>_data.npz
```

You can load it later via:

```python
import numpy as np
data = np.load("my_data.npz", allow_pickle=True)
```

---

## Requirements

- Python 3.10+
- `rclpy`
- `rosbag2_py`
- NumPy
- Matplotlib

(Optional)
- `mocap4r2_msgs` for mocap rigid body decoding

---

## Plotting

When `--plot` is enabled, the script generates:

### ✔ Joint positions  
6-subplot figure (one per UR joint)

### ✔ TCP pose  
6-subplot (x, y, z, roll, pitch, yaw)

### ✔ Mocap rigid body pose (if available)  
6-subplot (x, y, z, roll, pitch, yaw)

---
