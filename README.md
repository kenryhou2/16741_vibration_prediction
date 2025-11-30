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
From the scripts directory run the following:
```bash
Chirp 1: python3 bag_parse.py ../data/../../data/data_collection_11252025/chirp1/rosbag2_2025_11_25-19_24_21 --start-offset -0.98 --end-offset 0.10823 --plot --output data_chirp1_11292025.npz --ref-csv ../data/chirp1/ref_traj_chirp_clean.csv --linear 
Start offset: -2 + 1.08 = -0.92 
End Offset: 32.77067 - 31.8789 = 0.89177 = 1 – 0.89177= 0.10823 

Chirp 2: python3 bag_parse.py ../data/../../data/data_collection_11302025/chirp2/rosbag2_2025_11_30-02_02_18/ --start-offset -0.969 --end-offset 0.072 --plot --output data_chirp2_11302025.npz --ref-csv ../data/chirp2/ref_traj_chirp_clean.csv --linear 
Start offset: -2 + 1.031 = −0.969 
End Offset: 2 - (33.732 - 31.804) = 0.072 

AUX: Chirp3: python3 bag_parse.py ../data/../../data/data_collection_11302025/chirp3/rosbag2_2025_11_30-02_04_34/ --plot --output data_chirp3_11302025.npz --ref-csv ../data/chirp3/ref_traj_chirp3_clean_vicon_base.csv --start-offset .002 --end-offset 0.074 --aux-csv  
Start offset: -2 + 2.002 = .002 
End Offset: 2 - (34.728 - 32.802) = 0.074 

Cylinder1: python3 bag_parse.py ../data/../../data/data_collection_11302025/cylinder1/rosbag2_2025_11_30-02_09_09/ --plot --output data_cylinder1_11302025.npz --ref-csv ../data/cylinder1/ref_traj_cylinder_clean.csv --start-offset -0.9797 --end-offset 0.045 
Start offset: -2 + 1.0203 = −0.9797 
End offset: 2 – (17.56 - 15.605) = 0.045 

Cylinder2: python3 bag_parse.py ../data/../../data/data_collection_11252025/cylinder_run2/rosbag2_2025_11_25-19_14_51/ --plot --output data_cylinder2_11292025.npz --ref-csv ../data/cylinder2/ref_traj_cylinder_clean.csv --start-offset -0.7974 --end-offset 0.02559 
Start offset: -2 + 1.1471 + 0.0555 = -0.7974 
End offset: 2 - (17.675310 - 15.7009) = 0.02559 

Cylinder3: python3 bag_parse.py ../data/../../data/data_collection_11252025/cylinder_run3/rosbag2_2025_11_25-19_16_11/ --plot --output data_cylinder3_11292025.npz --ref-csv ../data/cylinder3/ref_traj_cylinder_clean.csv --start-offset 0.1545 --end-offset 0.005 
Start offset: -2 + 2.017 + 0.087 0.0505= 0.1545 
End offset: 2 - (18.56- 16.565) = 0.005 

Cylinder 4: python3 bag_parse.py ../data/../../data/data_collection_11302025/cylinder4/rosbag2_2025_11_30-02_36_13/ --plot --output data_cylinder4_11302025.npz --ref-csv ../data/cylinder4/ref_traj_cylinder_clean.csv --start-offset -0.8834 --end-offset 0.0557 
Start offset: –2 + 1.0286 + 0.088= −0.8834 
End offset: 2 – (17.5323 - 15.588) = 0.0557 

Linear1: python3 bag_parse.py ../data/../../data/data_collection_11252025/linear_run1/rosbag2_2025_11_25-19_27_33/ --plot --output data_linear1_11292025.npz --ref-csv ../data/linear1/ref_traj_linear_clean.csv --start-offset -0.9717 --end-offset 0.0103 --linear 
Start offset: -2 + 1.0283 = -0.9717 
End offset: 2 - (18.0571- 16.0674) = 0.0103 

AUX Linear 2: python3 bag_parse.py ../data/../../data/data_collection_11302025/linear2/rosbag2_2025_11_30-01_45_00/ --plot --output data_linear2_11302025.npz --ref-csv ../data/linear2/ref_traj_linear2_clean_vicon_base.csv --start-offset 0.0042 --end-offset 0.0976 --aux-csv 
Start offset: -2 + 2.0042 = 0.0042 
End offset: 2 - (19.0414- 17.139) = 0.0976 

AUX Linear 3: python3 bag_parse.py ../data/../../data/data_collection_11302025/linear3/rosbag2_2025_11_30-01_48_51/ --plot --output data_linear3_11302025.npz --ref-csv ../data/linear3/ref_traj_linear3_clean_vicon_base.csv --start-offset -0.9821 --end-offset 0.0638 --aux-csv 
Start offset: -2 + 1.0179 = −0.9821 
End offset: 2 - (18.0522- 16.116) = 0.0638 

AUX Roller Coaster1: python3 bag_parse.py ../data/../../data/data_collection_11302025/roller_coaster1/rosbag2_2025_11_30-02_52_14/ --plot --output data_rc1_11302025.npz --ref-csv ../data/rollercoaster1/ref_traj_rc1_clean_vicon_base.csv --start-offset -0.992 --end-offset 0.0986 --aux-csv 
Start offset: -2 + 1.008 = -0.992 
End offset: 2 - (31.4- 29.4986) = 0.0986 

AUX Roller Coaster2: python3 bag_parse.py ../data/../../data/data_collection_11302025/roller_coaster2/rosbag2_2025_11_30-03_01_39/ --plot --output data_rc2_11302025.npz --ref-csv ../data/rollercoaster2/ref_traj_rc2_clean_vicon_base.csv --start-offset 0 --end-offset 0 --aux-csv 
Start offset: -2 + 2 = 0 
End offset: 2 - (32.4360- 30.494) = 0.058 -> 0 

AUX Roller Coaster3: python3 bag_parse.py ../data/../../data/data_collection_11302025/roller_coaster3/rosbag2_2025_11_30-03_05_11/ --plot --output data_rc3_11302025.npz --ref-csv ../data/rollercoaster3/ref_traj_rc3_clean_vicon_base.csv --start-offset -0.9918 --end-offset 0.119 --aux-csv 
Start offset: -2 + 1.0082 = -0.9918 
End offset: 2 - (31.441- 29.56) = 0.119 

AUX Roller Coaster4: python3 bag_parse.py ../data/../../data/data_collection_11302025/roller_coaster4/rosbag2_2025_11_30-03_07_10/ --plot --output data_rc4_11302025.npz --ref-csv ../data/rollercoaster4/ref_traj_rc4_clean_vicon_base.csv --start-offset -0.992 --end-offset 0.028 --aux-csv 
Start offset: -2 + 1.008 = -0.992 
End offset: 2 - (31.415- 29.443) = 0.028 
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
