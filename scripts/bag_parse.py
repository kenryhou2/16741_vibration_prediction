#!/usr/bin/env python3
"""
Convert a rosbag2 into NumPy arrays for ML and plotting.

Signals extracted within a time window:
    [ first /waypoint_publisher/command  →  first /waypoint_publisher/status ]

Extract:
- /ur/joint_states (JointState)
- /ur/tcp_pose_current (PoseStamped)
- /ur/tcp_twist_current (TwistStamped)
- /rigid_bodies (RigidBodies.msg → rigidbodies[rigid_index].pose)

Saved into a single .npz with keys:
    Joint:
        time_rel
        ros_time_ns
        positions
        velocities
        efforts
        joint_names

    TCP pose:
        tcp_time_rel
        tcp_ros_time_ns
        tcp_position          (N,3)
        tcp_orientation_xyzw  (N,4)
        tcp_rpy               (N,3)

    TCP twist:
        tcp_twist_time_rel
        tcp_twist_ros_time_ns
        tcp_twist_linear      (N,3)
        tcp_twist_angular     (N,3)

    Rigid body pose (rigid_index):
        rb_time_rel
        rb_ros_time_ns
        rb_position           (N,3)
        rb_orientation_xyzw   (N,4)
        rb_rpy                (N,3)
        rb_name               (string)

Usage:
    python3 bag_to_joint_and_pose_arrays.py rosbag2_2025_11_25-19_24_21 \
        --joint-topic /ur/joint_states \
        --tcp-pose-topic /ur/tcp_pose_current \
        --tcp-twist-topic /ur/tcp_twist_current \
        --rigid-topic /rigid_bodies \
        --rigid-index 1 \
        --cmd-topic /waypoint_publisher/command \
        --status-topic /waypoint_publisher/status \
        --storage-id mcap \
        --output chirp1_joint_tcp_rb.npz \
        --plot
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import rosbag2_py


# =========================
# Basic utilities
# =========================

def open_bag_reader(bag_path: str, storage_id: str = "mcap") -> rosbag2_py.SequentialReader:
    if not os.path.isdir(bag_path):
        raise FileNotFoundError(f"Bag path '{bag_path}' is not a directory.")

    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id=storage_id,
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def get_topic_type_map(reader: rosbag2_py.SequentialReader) -> Dict[str, str]:
    topic_types = reader.get_all_topics_and_types()
    return {t.name: t.type for t in topic_types}


def extract_time_window_from_topics(
    bag_path: str,
    cmd_topic: str,
    status_topic: str,
    storage_id: str = "mcap",
) -> Tuple[int, int]:
    """
    Determine [start_ns, end_ns] using:
        start_ns  = first msg on cmd_topic
        end_ns    = first msg on status_topic
    """
    reader = open_bag_reader(bag_path, storage_id)

    start_ns = None
    end_ns = None

    while reader.has_next():
        topic, data, t_ns = reader.read_next()

        if start_ns is None and topic == cmd_topic:
            start_ns = t_ns

        if end_ns is None and topic == status_topic:
            end_ns = t_ns

        if start_ns is not None and end_ns is not None:
            break

    if start_ns is None:
        raise RuntimeError(f"No messages found on cmd_topic '{cmd_topic}' in bag '{bag_path}'.")
    if end_ns is None:
        raise RuntimeError(f"No messages found on status_topic '{status_topic}' in bag '{bag_path}'.")
    if end_ns <= start_ns:
        raise RuntimeError(
            f"Status time ({end_ns}) is not after command time ({start_ns}). "
            f"Check your topics or recording."
        )
    
    return start_ns, end_ns


# =========================
# JointState extraction
# =========================

def extract_joint_states_within_window(
    bag_path: str,
    joint_topic: str,
    start_ns: int,
    end_ns: int,
    storage_id: str = "mcap",
) -> Dict[str, np.ndarray]:
    reader = open_bag_reader(bag_path, storage_id)
    topic_type_map = get_topic_type_map(reader)

    if joint_topic not in topic_type_map:
        raise RuntimeError(f"Joint topic '{joint_topic}' not found. Available: {list(topic_type_map.keys())}")

    joint_type_str = topic_type_map[joint_topic]
    JointState = get_message(joint_type_str)

    times_ns: List[int] = []
    positions_list: List[List[float]] = []
    velocities_list: List[List[float]] = []
    efforts_list: List[List[float]] = []
    joint_names: List[str] = []

    first_joint_msg = True

    while reader.has_next():
        topic, data, t_ns = reader.read_next()

        if topic != joint_topic:
            continue

        if t_ns < start_ns or t_ns > end_ns:
            continue

        msg = deserialize_message(data, JointState)

        if first_joint_msg:
            joint_names = list(msg.name)
            first_joint_msg = False
        else:
            if list(msg.name) != joint_names:
                raise RuntimeError(
                    "JointState name ordering changed over time; script assumes fixed order."
                )

        times_ns.append(t_ns)
        positions_list.append(list(msg.position))
        velocities_list.append(list(msg.velocity))
        efforts_list.append(list(msg.effort))

    if not times_ns:
        raise RuntimeError(
            f"No joint_states messages on '{joint_topic}' between {start_ns} and {end_ns}."
        )

    times_ns = np.array(times_ns, dtype=np.int64)
    time_rel = (times_ns - start_ns).astype(np.float64) * 1e-9

    positions = np.array(positions_list, dtype=np.float64)
    velocities = np.array(velocities_list, dtype=np.float64)
    efforts = np.array(efforts_list, dtype=np.float64)
    joint_names_arr = np.array(joint_names, dtype=object)

    return {
        "time_rel": time_rel,
        "ros_time_ns": times_ns,
        "positions": positions,
        "velocities": velocities,
        "efforts": efforts,
        "joint_names": joint_names_arr,
    }


# =========================
# PoseStamped extraction
# =========================

def extract_pose_stamped_within_window(
    bag_path: str,
    pose_topic: str,
    start_ns: int,
    end_ns: int,
    storage_id: str = "mcap",
) -> Dict[str, np.ndarray]:
    """
    Extract geometry_msgs/PoseStamped-like messages:
        - position (x, y, z)
        - orientation (x, y, z, w)
    """
    reader = open_bag_reader(bag_path, storage_id)
    topic_type_map = get_topic_type_map(reader)

    if pose_topic not in topic_type_map:
        raise RuntimeError(f"Pose topic '{pose_topic}' not found in bag.")

    pose_type_str = topic_type_map[pose_topic]
    PoseStamped = get_message(pose_type_str)

    times_ns: List[int] = []
    positions_list: List[List[float]] = []
    orientations_list: List[List[float]] = []
    frame_ids: List[str] = []

    while reader.has_next():
        topic, data, t_ns = reader.read_next()

        if topic != pose_topic:
            continue

        if t_ns < start_ns or t_ns > end_ns:
            continue

        msg = deserialize_message(data, PoseStamped)

        p = msg.pose.position
        q = msg.pose.orientation
        positions_list.append([p.x, p.y, p.z])
        orientations_list.append([q.x, q.y, q.z, q.w])
        times_ns.append(t_ns)
        frame_ids.append(msg.header.frame_id)

    if not times_ns:
        raise RuntimeError(
            f"No pose messages on '{pose_topic}' between {start_ns} and {end_ns}."
        )

    times_ns = np.array(times_ns, dtype=np.int64)
    time_rel = (times_ns - start_ns).astype(np.float64) * 1e-9
    positions = np.array(positions_list, dtype=np.float64)
    orientations = np.array(orientations_list, dtype=np.float64)
    frame_ids_arr = np.array(frame_ids, dtype=object)

    rpy = quat_to_rpy_array(orientations)

    return {
        "time_rel": time_rel,
        "ros_time_ns": times_ns,
        "position": positions,
        "orientation_xyzw": orientations,
        "rpy": rpy,
        "frame_ids": frame_ids_arr,
    }


# =========================
# TwistStamped extraction
# =========================

def extract_twist_stamped_within_window(
    bag_path: str,
    twist_topic: str,
    start_ns: int,
    end_ns: int,
    storage_id: str = "mcap",
) -> Dict[str, np.ndarray]:
    """
    Extract geometry_msgs/TwistStamped-like messages:
        - linear (x, y, z)
        - angular (x, y, z)
    """
    reader = open_bag_reader(bag_path, storage_id)
    topic_type_map = get_topic_type_map(reader)

    if twist_topic not in topic_type_map:
        raise RuntimeError(f"Twist topic '{twist_topic}' not found in bag.")

    twist_type_str = topic_type_map[twist_topic]
    TwistStamped = get_message(twist_type_str)

    times_ns: List[int] = []
    linear_list: List[List[float]] = []
    angular_list: List[List[float]] = []
    frame_ids: List[str] = []

    while reader.has_next():
        topic, data, t_ns = reader.read_next()

        if topic != twist_topic:
            continue

        if t_ns < start_ns or t_ns > end_ns:
            continue

        msg = deserialize_message(data, TwistStamped)

        lin = msg.twist.linear
        ang = msg.twist.angular
        linear_list.append([lin.x, lin.y, lin.z])
        angular_list.append([ang.x, ang.y, ang.z])
        times_ns.append(t_ns)
        frame_ids.append(msg.header.frame_id)

    if not times_ns:
        raise RuntimeError(
            f"No twist messages on '{twist_topic}' between {start_ns} and {end_ns}."
        )

    times_ns = np.array(times_ns, dtype=np.int64)
    time_rel = (times_ns - start_ns).astype(np.float64) * 1e-9
    linear = np.array(linear_list, dtype=np.float64)
    angular = np.array(angular_list, dtype=np.float64)
    frame_ids_arr = np.array(frame_ids, dtype=object)

    return {
        "time_rel": time_rel,
        "ros_time_ns": times_ns,
        "linear": linear,
        "angular": angular,
        "frame_ids": frame_ids_arr,
    }


# =========================
# RigidBodies[rigid_index] extraction
# =========================

def quat_to_rpy(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    """
    Convert quaternion (x,y,z,w) to roll, pitch, yaw (XYZ / RPY) in radians.
    Standard aerospace / ROS convention.
    """
    # roll (x-axis)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.pi / 2 * np.sign(sinp)  # use 90 deg if out of range
    else:
        pitch = np.arcsin(sinp)

    # yaw (z-axis)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def quat_to_rpy_array(q_xyzw: np.ndarray) -> np.ndarray:
    """
    q_xyzw: (N,4) array of [x,y,z,w] quaternions.
    Returns (N,3) array of [roll, pitch, yaw].
    """
    rpy_list = []
    for q in q_xyzw:
        roll, pitch, yaw = quat_to_rpy(q[0], q[1], q[2], q[3])
        rpy_list.append([roll, pitch, yaw])
    return np.array(rpy_list, dtype=np.float64)


def extract_rigidbody_pose_within_window(
    bag_path: str,
    rigid_topic: str,
    rigid_index: int,
    start_ns: int,
    end_ns: int,
    storage_id: str = "mcap",
) -> Dict[str, np.ndarray]:
    """
    Extract pose of rigidbodies[rigid_index].pose from mocap4r2_msgs/msg/RigidBodies.

    RigidBodies.msg:
        std_msgs/Header header
        uint32 frame_number
        mocap4r2_msgs/RigidBody[] rigidbodies

    RigidBody.msg:
        string rigid_body_name
        Marker[] markers
        geometry_msgs/Pose pose
    """
    reader = open_bag_reader(bag_path, storage_id)
    topic_type_map = get_topic_type_map(reader)

    if rigid_topic not in topic_type_map:
        raise RuntimeError(f"RigidBodies topic '{rigid_topic}' not found in bag.")

    rb_type_str = topic_type_map[rigid_topic]
    RigidBodies = get_message(rb_type_str)

    times_ns: List[int] = []
    positions_list: List[List[float]] = []
    orientations_list: List[List[float]] = []
    frame_ids: List[str] = []
    rb_names: List[str] = []

    while reader.has_next():
        topic, data, t_ns = reader.read_next()

        if topic != rigid_topic:
            continue

        if t_ns < start_ns or t_ns > end_ns:
            continue

        msg = deserialize_message(data, RigidBodies)

        if rigid_index >= len(msg.rigidbodies):
            # Not enough rigid bodies in this frame; skip
            continue

        rb = msg.rigidbodies[rigid_index]
        pose = rb.pose
        p = pose.position
        q = pose.orientation

        positions_list.append([p.x, p.y, p.z])
        orientations_list.append([q.x, q.y, q.z, q.w])
        times_ns.append(t_ns)
        frame_ids.append(msg.header.frame_id)
        rb_names.append(rb.rigid_body_name)

    if not times_ns:
        raise RuntimeError(
            f"No RigidBody index {rigid_index} data found on '{rigid_topic}' "
            f"between {start_ns} and {end_ns}."
        )

    times_ns = np.array(times_ns, dtype=np.int64)
    time_rel = (times_ns - start_ns).astype(np.float64) * 1e-9
    positions = np.array(positions_list, dtype=np.float64)
    orientations = np.array(orientations_list, dtype=np.float64)
    frame_ids_arr = np.array(frame_ids, dtype=object)

    # Use the most frequent name as the label, in case it changes or is repeated
    unique_names, counts = np.unique(np.array(rb_names, dtype=object), return_counts=True)
    rb_name = unique_names[np.argmax(counts)]

    rpy = quat_to_rpy_array(orientations)

    return {
        "time_rel": time_rel,
        "ros_time_ns": times_ns,
        "position": positions,
        "orientation_xyzw": orientations,
        "rpy": rpy,
        "frame_ids": frame_ids_arr,
        "rb_name": np.array(rb_name, dtype=object),
    }


# =========================
# Query helpers for joints
# =========================

def get_joint_index(joint_names: np.ndarray, joint_name: str) -> int:
    for i, name in enumerate(joint_names.tolist()):
        if name == joint_name:
            return i
    raise ValueError(f"Joint name '{joint_name}' not found in {joint_names}.")


def get_joint_trajectory(data: Dict[str, np.ndarray], joint_name: str) -> Tuple[np.ndarray, np.ndarray]:
    idx = get_joint_index(data["joint_names"], joint_name)
    return data["time_rel"], data["positions"][:, idx]


def interpolate_joint_at_time(data: Dict[str, np.ndarray], joint_name: str, t_query: float) -> float:
    t = data["time_rel"]
    pos = data["positions"][:, get_joint_index(data["joint_names"], joint_name)]
    if t_query < t[0] or t_query > t[-1]:
        raise ValueError(f"t_query={t_query} is outside time range [{t[0]}, {t[-1]}]")
    return float(np.interp(t_query, t, pos))


# =========================
# Plotting helpers
# =========================

def plot_joint_positions(data: Dict[str, np.ndarray], title: str = "Joint Positions vs Time"):
    t = data["time_rel"]
    pos = data["positions"]
    joint_names = data["joint_names"].tolist()

    n_joints = pos.shape[1]
    n_rows, n_cols = 3, 2
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, figsize=(10, 8))
    fig.suptitle(title)

    for j in range(n_joints):
        row = j // n_cols
        col = j % n_cols
        ax = axes[row, col]
        ax.plot(t, pos[:, j])
        label = joint_names[j] if j < len(joint_names) else f"joint_{j}"
        ax.set_title(label)
        ax.set_ylabel("position [rad]")
        ax.grid(True)

    for j in range(n_joints, n_rows * n_cols):
        row = j // n_cols
        col = j % n_cols
        axes[row, col].axis("off")

    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.show()


def plot_pose_6d(time_rel: np.ndarray, pos: np.ndarray, rpy: np.ndarray, title: str):
    """
    Make 6 subplots:
        x, y, z, roll, pitch, yaw over time.
    pos: (N,3), rpy: (N,3)
    """
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(10, 8))
    fig.suptitle(title)

    labels = ["x [m]", "y [m]", "z [m]", "roll [rad]", "pitch [rad]", "yaw [rad]"]
    data_series = [pos[:, 0], pos[:, 1], pos[:, 2], rpy[:, 0], rpy[:, 1], rpy[:, 2]]

    for i in range(6):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        ax.plot(time_rel, data_series[i])
        ax.set_ylabel(labels[i])
        ax.grid(True)

    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.show()


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(description="Extract joint, TCP, and mocap pose data from rosbag2 into NumPy arrays.")
    parser.add_argument("bag_path", help="Path to rosbag2 directory (e.g., rosbag2_2025_11_25-19_24_21)")
    parser.add_argument("--joint-topic", default="/ur/joint_states",
                        help="JointState topic (default: /ur/joint_states)")
    parser.add_argument("--tcp-pose-topic", default="/ur/tcp_pose_current",
                        help="TCP PoseStamped topic (default: /ur/tcp_pose_current)")
    parser.add_argument("--tcp-twist-topic", default="/ur/tcp_twist_current",
                        help="TCP TwistStamped topic (default: /ur/tcp_twist_current)")
    parser.add_argument("--rigid-topic", default="/rigid_bodies",
                        help="RigidBodies topic (default: /rigid_bodies)")
    parser.add_argument("--rigid-index", type=int, default=1,
                        help="Index of rigid body in RigidBodies.rigidbodies array (default: 1)")
    parser.add_argument("--cmd-topic", default="/waypoint_publisher/command",
                        help="Topic defining start time (first message) (default: /waypoint_publisher/command)")
    parser.add_argument("--status-topic", default="/waypoint_publisher/status",
                        help="Topic defining end time (first message) (default: /waypoint_publisher/status)")
    parser.add_argument("--storage-id", default="mcap",
                        help="rosbag2 storage id (e.g., mcap, sqlite3). Default: mcap")
    parser.add_argument("--output", "-o", default=None,
                        help="Output .npz file path (default: <bag_dir>_data.npz)")
    parser.add_argument("--plot", action="store_true",
                        help="If set, plot joint positions, rigid body and TCP pose.")
    parser.add_argument("--start-offset", type=float, default=0.0,
                    help="Seconds to subtract from start of time window (can be negative).")
    parser.add_argument("--end-offset", type=float, default=0.0,
                        help="Seconds to add to end of time window (can be negative).")


    args = parser.parse_args()

    rclpy.init(args=None)
    try:
        print(f"[INFO] Determining time window from '{args.cmd_topic}' → '{args.status_topic}'...")
        start_ns, end_ns = extract_time_window_from_topics(
            args.bag_path,
            cmd_topic=args.cmd_topic,
            status_topic=args.status_topic,
            storage_id=args.storage_id,
        )
        # Apply manual offsets
        start_ns_offset = start_ns + int(args.start_offset * 1e9)
        end_ns_offset   = end_ns   + int(args.end_offset   * 1e9)

        print(f"[INFO] Applying offsets: start_offset={args.start_offset}s, end_offset={args.end_offset}s")
        print(f"[INFO] Adjusted window: start={start_ns_offset}, end={end_ns_offset}")

        # Ensure the window is valid
        if end_ns_offset <= start_ns_offset:
            raise RuntimeError(
                f"After offsets, end time ({end_ns_offset}) is not after start time ({start_ns_offset})."
            )

        # Use the adjusted window for extraction
        start_ns = start_ns_offset
        end_ns   = end_ns_offset
        
        dt = (end_ns - start_ns) * 1e-9
        print(f"[INFO] Time window: start={start_ns} ns, end={end_ns} ns, duration={dt:.3f} s")

        # --- Joint states ---
        print(f"[INFO] Extracting joint states from '{args.joint_topic}'...")
        joint_data = extract_joint_states_within_window(
            args.bag_path,
            joint_topic=args.joint_topic,
            start_ns=start_ns,
            end_ns=end_ns,
            storage_id=args.storage_id,
        )

        all_data = {
            "time_rel": joint_data["time_rel"],
            "ros_time_ns": joint_data["ros_time_ns"],
            "positions": joint_data["positions"],
            "velocities": joint_data["velocities"],
            "efforts": joint_data["efforts"],
            "joint_names": joint_data["joint_names"],
        }

        # --- TCP pose ---
        tcp_pose_data = None
        try:
            print(f"[INFO] Extracting TCP pose from '{args.tcp_pose_topic}'...")
            tcp_pose_data = extract_pose_stamped_within_window(
                args.bag_path,
                pose_topic=args.tcp_pose_topic,
                start_ns=start_ns,
                end_ns=end_ns,
                storage_id=args.storage_id,
            )
            all_data.update({
                "tcp_time_rel": tcp_pose_data["time_rel"],
                "tcp_ros_time_ns": tcp_pose_data["ros_time_ns"],
                "tcp_position": tcp_pose_data["position"],
                "tcp_orientation_xyzw": tcp_pose_data["orientation_xyzw"],
                "tcp_rpy": tcp_pose_data["rpy"],
                "tcp_frame_ids": tcp_pose_data["frame_ids"],
            })
        except RuntimeError as e:
            print(f"[WARN] TCP pose extraction skipped: {e}")

        # --- TCP twist ---
        tcp_twist_data = None
        try:
            print(f"[INFO] Extracting TCP twist from '{args.tcp_twist_topic}'...")
            tcp_twist_data = extract_twist_stamped_within_window(
                args.bag_path,
                twist_topic=args.tcp_twist_topic,
                start_ns=start_ns,
                end_ns=end_ns,
                storage_id=args.storage_id,
            )
            all_data.update({
                "tcp_twist_time_rel": tcp_twist_data["time_rel"],
                "tcp_twist_ros_time_ns": tcp_twist_data["ros_time_ns"],
                "tcp_twist_linear": tcp_twist_data["linear"],
                "tcp_twist_angular": tcp_twist_data["angular"],
                "tcp_twist_frame_ids": tcp_twist_data["frame_ids"],
            })
        except RuntimeError as e:
            print(f"[WARN] TCP twist extraction skipped: {e}")

        # --- Rigid body pose (index) ---
        rb_data = None
        try:
            print(f"[INFO] Extracting RigidBody index {args.rigid_index} from '{args.rigid_topic}'...")
            rb_data = extract_rigidbody_pose_within_window(
                args.bag_path,
                rigid_topic=args.rigid_topic,
                rigid_index=args.rigid_index,
                start_ns=start_ns,
                end_ns=end_ns,
                storage_id=args.storage_id,
            )
            all_data.update({
                "rb_time_rel": rb_data["time_rel"],
                "rb_ros_time_ns": rb_data["ros_time_ns"],
                "rb_position": rb_data["position"],
                "rb_orientation_xyzw": rb_data["orientation_xyzw"],
                "rb_rpy": rb_data["rpy"],
                "rb_frame_ids": rb_data["frame_ids"],
                "rb_name": rb_data["rb_name"],
            })
        except RuntimeError as e:
            print(f"[WARN] RigidBody extraction skipped: {e}")

        # --- Save ---
        out_path = args.output
        if out_path is None:
            base = os.path.basename(args.bag_path.rstrip("/"))
            out_path = os.path.abspath(base + "_data.npz")

        np.savez(out_path, **all_data)
        print(f"[INFO] Saved data to: {out_path}")
        print(f"[INFO] Keys: {list(all_data.keys())}")
        print(f"[INFO] Joint names: {joint_data['joint_names']}")

        # Quick elbow sanity check (optional)
        for candidate in ["elbow_joint", "ur_elbow_joint"]:
            try:
                t_elbow, elbow = get_joint_trajectory(joint_data, candidate)
                print(f"[INFO] Example elbow trajectory for joint '{candidate}': {elbow.shape[0]} samples.")
                break
            except ValueError:
                continue

        # --- Plotting ---
        if args.plot:
            plot_joint_positions(joint_data, title="Joint Positions (cmd → status window)")

            if rb_data is not None:
                rb_name = rb_data["rb_name"].item()
                title_rb = f"Rigid body {args.rigid_index} pose ({rb_name})"
                plot_pose_6d(rb_data["time_rel"], rb_data["position"], rb_data["rpy"], title=title_rb)

            if tcp_pose_data is not None:
                title_tcp = "TCP pose (cmd → status window)"
                plot_pose_6d(tcp_pose_data["time_rel"], tcp_pose_data["position"], tcp_pose_data["rpy"], title=title_tcp)

    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
