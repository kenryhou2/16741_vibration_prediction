# ROS2 Bag TF Tree reconstruction:
Usage: From the scripts directory,
```python3 bag_tf.py --bag ../../data/data_collection_11252025/chirp1/rosbag2_2025_11_25-19_24_21/ --source-frame surface --target-frame vicon_base --tf-time-offset```

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
## Usage notes per data run:
```bash
Data offsets; 

Chirp 1: hkou@UBE-CARBON:~/Courses/16741_Manipulation_Erickson/project/16741_vibration_prediction/scripts$ python3 bag_parse.py ../data/../../data/data_collection_11252025/chirp1/rosbag2_2025_11_25-19_24_21 --start-offset -0.98 --end-offset 0.10823 --plot --output data_chirp1_11292025.npz --ref-csv ../data/chirp1/ref_traj_chirp_clean.csv --linear 

Start offset: -2 + 1.08 = -0.92 

End Offset: 32.77067 - 31.8789 = 0.89177 = 1 – 0.89177= 0.10823 

 

 

[Deprecated] Cylinder1: hkou@UBE-CARBON:~/Courses/16741_Manipulation_Erickson/project/16741_vibration_prediction/scripts$ python3 bag_parse.py ../data/../../data/data_collection_11252025/cylinder_run1/rosbag2_2025_11_25-18_47_41/ --plot --output data_cylinder_11292025.npz --ref-csv ../data/cylinder1/ref_traj_cylinder_clean.csv --start-offset -0.92 --end-offset 0.02233  

Start offset: –2 + 1 = -1 

End offset: 2 – 1.97767 = 0.02233 

17.57887 - 15.6012 = 1.97767 

 

 Cylinder2: hkou@UBE-CARBON:~/Courses/16741_Manipulation_Erickson/project/16741_vibration_prediction/scripts$ python3 bag_parse.py ../data/../../data/data_collection_11252025/cylinder_run2/rosbag2_2025_11_25-19_14_51/ --plot --output data_cylinder2_11292025.npz --ref-csv ../data/cylinder2/ref_traj_cylinder_clean.csv --start-offset -0.7974 --end-offset 0.02559 

Start offset: -2 + 1.1471 + 0.0555 = -0.7974 

End offset: 2 - (17.675310 - 15.7009) = 0.02559 

 

Cylinder3: python3 bag_parse.py ../data/../../data/data_collection_11252025/cylinder_run3/rosbag2_2025_11_25-19_16_11/ --plot --output data_cylinder3_11292025.npz --ref-csv ../data/cylinder3/ref_traj_cylinder_clean.csv --start-offset 0.1545 --end-offset 0.005 

Start offset: -2 + 2.017 + 0.087 0.0505= 0.1545 

End offset: 2 - (18.56- 16.565) = 0.005 

 

Linear1: python3 bag_parse.py ../data/../../data/data_collection_11252025/linear_run1/rosbag2_2025_11_25-19_27_33/ --plot --output data_linear1_11292025.npz --ref-csv ../data/linear1/ref_traj_linear_clean.csv --start-offset -0.9717 --end-offset 0.0103 --linear 

Start offset: -2 + 1.0283 = -0.9717 

End offset: 2 - (18.0571- 16.0674) = 0.0103 
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
