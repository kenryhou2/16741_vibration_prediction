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
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import CubicSpline
from bag_tf import BagTFGraph, _get_tf_message_at_offset, _transform_to_matrix
import rosbag2_py


# === Static fallback transforms (from YAML) ===

# vicon_base_to_UR_base (YAML): transform from vicon_base -> UR_base
_FALLBACK_VICON_BASE_TO_UR_BASE_QUAT = [0.007095, 0.005023, -0.999812, 0.017318]  # [x,y,z,w]
_FALLBACK_VICON_BASE_TO_UR_BASE_TRANS = [-0.35790947, 0.10588769, -0.00464215]

# Default ideal UR_base -> surface (chirp)
_FALLBACK_UR_BASE_TO_SURFACE_CHIRP_QUAT = [0.002, -0.005, 1.000, 0.015]
_FALLBACK_UR_BASE_TO_SURFACE_CHIRP_TRANS = [-0.592, 0.084, 0.059]

# Linear-case ideal UR_base -> surface
_FALLBACK_UR_BASE_TO_SURFACE_LINEAR_QUAT = [-0.006, -0.017, 1.000, -0.013]
_FALLBACK_UR_BASE_TO_SURFACE_LINEAR_TRANS = [-0.569, -0.017, 0.058]


def _make_T_from_quat_trans(quat, trans):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = np.array(trans, dtype=float)
    return T



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
# Reference trajectory loading and interpolation
# =========================

def load_and_interpolate_reference_trajectory(
    csv_path: str,
    target_time_rel: np.ndarray,
    *,
    bag_path: Optional[str] = None,
    tf_time_offset: Optional[float] = 1.0,
    vicon_base_frame: str = "vicon_base",
    ur_base_frame: str = "UR_base",
    surface_frame: str = "surface",
    ideal_surface_frame: Optional[str] = None,
    linear_surface: bool = False,  
) -> Dict[str, np.ndarray]:
    """
    Load reference trajectory from CSV, optionally transform it using TF from a bag,
    then interpolate onto target_time_rel.

    TF usage (if bag_path is provided):
        - Preferred path:
            1) Build TF graph from bag (static + /tf snapshot).
            2) Extract:
                vicon_base_to_UR_base   = TF(ur_base_frame -> vicon_base_frame)
                vicon_base_to_surface   = TF(surface_frame -> vicon_base_frame)
                UR_base_to_actual_surface =
                    inv(vicon_base_to_UR_base) @ vicon_base_to_surface   # same as current code

            3) Ideal surface:
                - If linear_surface is True:
                    use hardcoded "linear" UR_base_to_surface.
                - Else if ideal_surface_frame is given:
                    UR_base_to_ideal_surface = TF(ideal_surface_frame -> ur_base_frame)
                - Else:
                    use hardcoded "chirp" UR_base_to_surface.

                ideal_surface_to_UR_base = inv(UR_base_to_ideal_surface)

        - Failsafe path (e.g. no /tf_static):
            1) vicon_base_to_UR_base from hardcoded YAML (vicon -> UR),
               then invert to match variable semantics.
            2) vicon_base_to_surface from /tf snapshot at tf_time_offset.
            3) UR_base_to_actual_surface = inv(vicon_base_to_UR_base) @ vicon_base_to_surface.
            4) UR_base_to_ideal_surface (static) from either chirp or linear.
               ideal_surface_to_UR_base = inv(UR_base_to_ideal_surface).

        For each CSV pose, build T from (pos, quat) and apply:
            new_T = UR_base_to_actual_surface @ ideal_surface_to_UR_base @ T

    Interpolation:
        - Parameterize the (transformed) reference samples by alpha in [0,1].
        - Fit cubic splines x(alpha), y(alpha), z(alpha).
        - Map target_time_rel linearly into [0,1] and sample splines there.

    Returns:
        {
            "ref_time_rel": target_time_rel,
            "ref_position": (N,3) array,  # transformed & interpolated XYZ
        }
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Reference CSV '{csv_path}' not found.")

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.dtype.names is None:
        raise RuntimeError(
            f"CSV '{csv_path}' has no header row; please add one."
        )

    names = data.dtype.names
    M = len(data)
    if M < 2:
        raise RuntimeError(f"Reference CSV '{csv_path}' must have at least 2 samples, got {M}.")

    # --- Position columns (your file: x, y, z, px, py, pz, pw, u) ---
    pos_candidates = [
        ("x", "y", "z"),
        ("px", "py", "pz"),  # fallback
    ]
    pos_cols = None
    for cand in pos_candidates:
        if all(c in names for c in cand):
            pos_cols = cand
            break
    if pos_cols is None:
        raise RuntimeError(f"Could not find any position columns among {names}")

    pos_ref = np.vstack([np.asarray(data[c], dtype=np.float64) for c in pos_cols]).T  # (M,3)

    # --- Quaternion columns (if present) ---
    quat_cols = None
    quat_candidates = [
        ("px", "py", "pz", "pw"),
        ("qx", "qy", "qz", "qw"),
        ("ox", "oy", "oz", "ow"),
    ]
    for cand in quat_candidates:
        if all(c in names for c in cand):
            quat_cols = cand
            break

    quat_ref = None
    if quat_cols is not None:
        quat_ref = np.vstack([np.asarray(data[c], dtype=np.float64) for c in quat_cols]).T  # (M,4)

    # ------------------------------------------------------------------
    # 1) Build UR_base_to_actual_surface & ideal_surface_to_UR_base
    # ------------------------------------------------------------------
    UR_base_to_actual_surface = None
    ideal_surface_to_UR_base = None

    if bag_path is not None:
        try:
            # Preferred: use full TF graph
            tf_graph = BagTFGraph.from_bag(
                bag_path,
                tf_static_topic="/tf_static",
                tf_topic="/tf",
                tf_time_offset=tf_time_offset,
            )

            # vicon_base <- ur_base, vicon_base <- surface  (same as your current code)
            vicon_base_to_UR_base = tf_graph.lookup_transform(ur_base_frame, vicon_base_frame)
            vicon_base_to_surface = tf_graph.lookup_transform(surface_frame, vicon_base_frame)
            UR_base_to_surface = tf_graph.lookup_transform(surface_frame, ur_base_frame)
            # Print for debugging
            R_vicon_to_UR = vicon_base_to_UR_base[:3, :3]
            q_vicon_to_UR = R.from_matrix(R_vicon_to_UR).as_quat()
            t_vicon_to_UR = vicon_base_to_UR_base[:3, 3]
            R_vicon_to_surf = vicon_base_to_surface[:3, :3]
            q_vicon_to_surf = R.from_matrix(R_vicon_to_surf).as_quat()
            t_vicon_to_surf = vicon_base_to_surface[:3, 3]
            print(f"TF {vicon_base_frame} <- {ur_base_frame}:")
            print(f" q={q_vicon_to_UR}")
            print(f" t={t_vicon_to_UR}")
            print(f"TF {vicon_base_frame} <- {surface_frame}:")
            print(f" q={q_vicon_to_surf}")
            print(f" t={t_vicon_to_surf}")

            # UR_base -> actual_surface (same expression you already use)
            UR_base_to_actual_surface = np.linalg.inv(vicon_base_to_UR_base) @ vicon_base_to_surface

            # Ideal surface: choose static or TF depending on flag
            if linear_surface:
                # Linear hardcoded ideal UR_base -> surface
                UR_base_to_surface_static = _make_T_from_quat_trans(
                    _FALLBACK_UR_BASE_TO_SURFACE_LINEAR_QUAT,
                    _FALLBACK_UR_BASE_TO_SURFACE_LINEAR_TRANS,
                )
                UR_base_to_ideal_surface = UR_base_to_surface_static
                
            else:
                #cylindrical
                if ideal_surface_frame is None:
                    ideal_surface_frame = surface_frame

                # UR_base -> ideal_surface and its inverse
                try:
                    
                    UR_base_to_ideal_surface = tf_graph.lookup_transform(ideal_surface_frame, ur_base_frame)
                    ideal_surface_to_UR_base = np.linalg.inv(UR_base_to_ideal_surface)
                except Exception as e_tf_ideal:
                    print(f"[WARN] TF lookup for ideal_surface_frame '{ideal_surface_frame}' failed ({e_tf_ideal}). "
                          f"Falling back to CHIRP static UR_base -> surface.")                
                    UR_base_to_surface_static = _make_T_from_quat_trans(
                        _FALLBACK_UR_BASE_TO_SURFACE_CHIRP_QUAT,
                        _FALLBACK_UR_BASE_TO_SURFACE_CHIRP_TRANS,
                    )
                    UR_base_to_ideal_surface = UR_base_to_surface_static
            
            print("surface to UR_base:")
            R_UR_base_to_ideal = UR_base_to_ideal_surface[:3, :3]
            q_UR_base_to_ideal = R.from_matrix(R_UR_base_to_ideal).as_quat()
            t_UR_base_to_ideal = UR_base_to_ideal_surface[:3, 3]
            print(f"TF {ideal_surface_frame} -> {ur_base_frame}:")
            print(f" q={q_UR_base_to_ideal}")
            print(f" t={t_UR_base_to_ideal}")  

            ideal_surface_to_UR_base = np.linalg.inv(UR_base_to_ideal_surface)
            # R_UR_base_to_ideal = UR_base_to_ideal_surface[:3, :3]
            # q_UR_base_to_ideal = R.from_matrix(R_UR_base_to_ideal).as_quat()
            # t_UR_base_to_ideal = UR_base_to_ideal_surface[:3, 3]
            # print(f"TF {ur_base_frame} -> {ideal_surface_frame}:")
            # print(f" q={q_UR_base_to_ideal}")
            # print(f" t={t_UR_base_to_ideal}")

        except Exception as e:
            # For tf_static missing.
            print(f"[WARN] TF_static missing: TF graph construction / lookup failed ({e}).")
            
           
    # ------------------------------------------------------------------
    # 2) Apply transforms to the CSV poses
    # ------------------------------------------------------------------
    pos_tf = np.zeros_like(pos_ref)
    quat_tf = np.zeros_like(quat_ref) if quat_ref is not None else None

    for i in range(M):
        T = np.eye(4)
        T[:3, 3] = pos_ref[i]

        if quat_ref is not None:
            T[:3, :3] = R.from_quat(quat_ref[i]).as_matrix()
        else:
            T[:3, :3] = np.eye(3)

        # new_T = UR_base_to_actual_surface @ ideal_surface_to_UR_base @ T
        new_T = UR_base_to_actual_surface @ ideal_surface_to_UR_base @ T

        pos_tf[i] = new_T[:3, 3]
        if quat_tf is not None:
            quat_tf[i] = R.from_matrix(new_T[:3, :3]).as_quat()

    # Use transformed positions as the reference for interpolation
    pos_ref = pos_tf

    # ------------------------------------------------------------------
    # 3) Interpolate transformed positions onto target_time_rel
    # ------------------------------------------------------------------
    alpha_ref = np.linspace(0.0, 1.0, M, dtype=np.float64)

    t_min = float(target_time_rel[0])
    t_max = float(target_time_rel[-1])
    if t_max <= t_min:
        raise RuntimeError(
            f"target_time_rel must be strictly increasing; got t_min={t_min}, t_max={t_max}"
        )

    alpha_query = (target_time_rel - t_min) / (t_max - t_min)

    cs_x = CubicSpline(alpha_ref, pos_ref[:, 0], extrapolate=False)
    cs_y = CubicSpline(alpha_ref, pos_ref[:, 1], extrapolate=False)
    cs_z = CubicSpline(alpha_ref, pos_ref[:, 2], extrapolate=False)

    x_interp = cs_x(alpha_query)
    y_interp = cs_y(alpha_query)
    z_interp = cs_z(alpha_query)
    pos_interp = np.vstack([x_interp, y_interp, z_interp]).T  # (N,3)

    return {
        "ref_time_rel": target_time_rel,
        "ref_position": pos_interp,
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


def plot_pose_6d(
    time_rel: np.ndarray,
    pos: np.ndarray,
    rpy: np.ndarray,
    title: str,
    ref_pos: np.ndarray = None,
):
    """
    Make 6 subplots:
        x, y, z, roll, pitch, yaw over time.

    pos: (N,3)  - TCP position
    rpy: (N,3)  - TCP roll, pitch, yaw
    ref_pos: (N,3) or None - reference position to overlay on x,y,z
    """
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(10, 8))
    fig.suptitle(title)

    labels = ["x [m]", "y [m]", "z [m]", "roll [rad]", "pitch [rad]", "yaw [rad]"]
    data_series = [pos[:, 0], pos[:, 1], pos[:, 2], rpy[:, 0], rpy[:, 1], rpy[:, 2]]

    for i in range(6):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        # Always plot TCP
        ax.plot(time_rel, data_series[i], label="TCP")

        # For the first three (x,y,z), optionally overlay reference
        if ref_pos is not None and i < 3:
            ax.plot(time_rel, ref_pos[:, i], linestyle="--", label="Reference")

        ax.set_ylabel(labels[i])
        ax.grid(True)

        # Only put legend on the first subplot if ref is present
        if ref_pos is not None and i == 0:
            ax.legend()

    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.show()


def plot_pose_overlay_6d(
    time_rel: np.ndarray,
    tcp_pos: np.ndarray,
    tcp_rpy: np.ndarray,
    ref_pos: np.ndarray,
    ref_rpy: np.ndarray,
    title: str = "TCP vs Reference Pose"
):
    """
    Overlay TCP pose and reference pose in 6 subplots:
        x, y, z, roll, pitch, yaw vs time.

    All arrays are shape (N,3), time_rel is (N,).
    """
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(10, 8))
    fig.suptitle(title)

    labels = ["x [m]", "y [m]", "z [m]", "roll [rad]", "pitch [rad]", "yaw [rad]"]

    tcp_series = [
        tcp_pos[:, 0],
        tcp_pos[:, 1],
        tcp_pos[:, 2],
        tcp_rpy[:, 0],
        tcp_rpy[:, 1],
        tcp_rpy[:, 2],
    ]

    ref_series = [
        ref_pos[:, 0],
        ref_pos[:, 1],
        ref_pos[:, 2],
        ref_rpy[:, 0],
        ref_rpy[:, 1],
        ref_rpy[:, 2],
    ]

    for i in range(6):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        ax.plot(time_rel, tcp_series[i], label="TCP")
        ax.plot(time_rel, ref_series[i], linestyle="--", label="Reference")
        ax.set_ylabel(labels[i])
        ax.grid(True)

        if row == 0 and col == 0:
            ax.legend()

    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.show()

def plot_tcp_ref_delta(
    time_rel: np.ndarray,
    delta_pos: np.ndarray,
    title: str = "Reference - TCP Position Error"
):
    """
    Plot the delta between TCP position and reference position
    for each axis x, y, z in a separate figure with 3 subplots.

    delta_pos: (N,3) array where columns are [dx, dy, dz] = tcp - ref
    """
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    fig.suptitle(title)

    labels = [r"Δx [m] (tcp - ref)", r"Δy [m] (tcp - ref)", r"Δz [m] (tcp - ref)"]

    for i in range(3):
        ax = axes[i]
        ax.plot(time_rel, delta_pos[:, i])
        ax.axhline(0.0, linestyle="--", linewidth=0.8)
        ax.set_ylabel(labels[i])
        ax.grid(True)

    axes[-1].set_xlabel("time [s]")
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
    parser.add_argument("--ref-csv", default=None,
                        help="Optional CSV file with reference trajectory (t, x, y, z, quaternion wxyz). "
                             "Will be interpolated to the TCP time grid.")
    parser.add_argument("--linear", action="store_true",
                        help="Use hardcoded linear UR_base_to_surface static transform instead of default chirp surface.")



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

            # --- Reference trajectory  ---
            if args.ref_csv is not None:
                print(f"[INFO] Loading and transforming + interpolating reference trajectory from '{args.ref_csv}'...")
                ref_data = load_and_interpolate_reference_trajectory(
                    args.ref_csv,
                    target_time_rel=tcp_pose_data["time_rel"],
                    bag_path=args.bag_path,
                    tf_time_offset=1.0,
                    vicon_base_frame="vicon_base",
                    ur_base_frame="UR_base",
                    surface_frame="surface",
                    ideal_surface_frame=None,
                    linear_surface=args.linear,  
                )
                all_data.update(ref_data)
                print(f"[INFO] Reference position shape: {ref_data['ref_position'].shape}")

                # --- Delta between reference and TCP pose (per-axis) ---
                if tcp_pose_data is not None:
                    tcp_pos = all_data["tcp_position"]       # (N,3)
                    ref_pos = all_data["ref_position"]       # (N,3)

                    # Be defensive in case of any subtle length mismatch
                    N = min(tcp_pos.shape[0], ref_pos.shape[0])
                    if N == 0:
                        print("[WARN] No overlap between TCP and reference for delta computation.")
                    else:
                        # Δ = tcp - ref for each axis
                        delta_pos = tcp_pos[:N, :] - ref_pos[:N, :]
                        all_data["tcp_ref_delta_position"] = delta_pos
                        print(f"[INFO] Delta (tcp - ref) position shape: {delta_pos.shape}")

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
                # If we have a reference trajectory, pass its positions; otherwise None
                ref_pos = all_data["ref_position"] if "ref_position" in all_data else None
                title_tcp = "TCP pose (cmd → status window)"
                plot_pose_6d(
                    tcp_pose_data["time_rel"],
                    tcp_pose_data["position"],
                    tcp_pose_data["rpy"],
                    title=title_tcp,
                    ref_pos=ref_pos,
                )
                if "tcp_ref_delta_position" in all_data:
                    delta_pos = all_data["tcp_ref_delta_position"]
                    # Use matching time vector (truncate if needed)
                    N_delta = delta_pos.shape[0]
                    t_delta = tcp_pose_data["time_rel"][:N_delta]
                    plot_tcp_ref_delta(
                        t_delta,
                        delta_pos,
                        title="(TCP Position - Reference) Error (cmd → status window)",
                    )

    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
