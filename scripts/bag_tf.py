#!/usr/bin/env python3
"""
bag_tf_graph.py

Utilities to build a TF tree from a ROS2 bag (static + optional /tf snapshot)
and query transforms between frames.

Usage as a library:

    from bag_tf_graph import BagTFGraph
    import numpy as np

    bag_path = "/path/to/rosbag2_2025_11_25-19_24_21"

    tf_graph = BagTFGraph.from_bag(
        bag_path,
        tf_static_topic="/tf_static",
        tf_topic="/tf",
        tf_time_offset=1.0,  # seconds after first /tf
    )

    # Example: get vicon_base -> UR_base and vicon_base -> surface
    T_vicon_to_UR = tf_graph.lookup_transform("vicon_base", "UR_base")
    T_vicon_to_surface = tf_graph.lookup_transform("vicon_base", "surface")

    # Then reconstruct UR_base -> surface
    T_UR_to_surface = np.linalg.inv(T_vicon_to_UR) @ T_vicon_to_surface

You can also run this file directly for debugging to dump the TF tree and
a specific transform:

    python3 bag_tf_graph.py --bag /path/to/bag \
        --source-frame vicon_base --target-frame surface \
        --tf-time-offset 1.0
"""

import argparse
import os
from collections import defaultdict, deque

import numpy as np
from scipy.spatial.transform import Rotation as R
import tf2_ros as tf2
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message


# -------------------------------------------------------------------
# Low-level helpers
# -------------------------------------------------------------------

def _transform_to_matrix(transform_msg):
    """
    Convert geometry_msgs/Transform to a 4x4 homogeneous matrix.

    TF convention: this transform is the pose of the CHILD in the PARENT frame.
    It maps CHILD-frame coordinates -> PARENT-frame coordinates.
    """
    tx = transform_msg.translation.x
    ty = transform_msg.translation.y
    tz = transform_msg.translation.z

    qx = transform_msg.rotation.x
    qy = transform_msg.rotation.y
    qz = transform_msg.rotation.z
    qw = transform_msg.rotation.w

    T = np.eye(4)
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = np.array([tx, ty, tz], dtype=float)
    return T


def _add_tfmessage_to_graph(tf_msg, neighbors, children):
    """
    Add all transforms from a tf2_msgs/TFMessage to the TF graph.

    neighbors: dict frame -> list[(neighbor_frame, T_frame_to_neighbor)]
               where T_frame_to_neighbor maps coords in 'frame' -> 'neighbor_frame'.
    children:  dict parent_frame -> list[child_frame] for tree printing.
    """
    for ts in tf_msg.transforms:
        parent = ts.header.frame_id.lstrip('/')
        child = ts.child_frame_id.lstrip('/')

        # T_child_parent: child coords -> parent coords (child pose in parent frame)
        T_child_parent = _transform_to_matrix(ts.transform)
        # For graph, we want both directions:
        # parent -> child  AND  child -> parent
        T_parent_child = np.linalg.inv(T_child_parent)

        neighbors[parent].append((child, T_parent_child))   # parent -> child
        neighbors[child].append((parent, T_child_parent))   # child -> parent

        children[parent].append(child)
        _ = children[child]  # ensure key exists


def _read_static_tf_from_bag(bag_path, tf_static_topic="/tf_static"):
    """
    Build TF graph from /tf_static topic in the bag.

    Returns:
        neighbors, children
    """
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topics_and_types = reader.get_all_topics_and_types()
    topic_type_map = {t.name: t.type for t in topics_and_types}

    if tf_static_topic not in topic_type_map:
        raise RuntimeError(f"No {tf_static_topic} topic found in bag.")

    TFMessage = get_message(topic_type_map[tf_static_topic])

    neighbors = defaultdict(list)
    children = defaultdict(list)

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic != tf_static_topic:
            continue

        msg = deserialize_message(data, TFMessage)
        _add_tfmessage_to_graph(msg, neighbors, children)

    return neighbors, children


def _get_tf_message_at_offset(bag_path, tf_topic="/tf", time_offset_s=1.0):
    """
    Return a single TFMessage from /tf at approximately
    (first_tf_time + time_offset_s).

    Uses bag timestamps from rosbag2_py (nanoseconds since epoch):
      - first_tf_time = time of first /tf msg.
      - target_time   = first_tf_time + time_offset_s * 1e9
      - returns last /tf msg with t <= target_time,
        or first /tf msg if that is already after target_time.

    Raises:
        RuntimeError if no /tf messages are found.
    """
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topics_and_types = reader.get_all_topics_and_types()
    topic_type_map = {t.name: t.type for t in topics_and_types}

    if tf_topic not in topic_type_map:
        raise RuntimeError(f"No {tf_topic} topic found in bag.")

    TFMessage = get_message(topic_type_map[tf_topic])

    first_tf_time = None
    target_time = None
    last_msg = None
    last_time = None

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic != tf_topic:
            continue

        if first_tf_time is None:
            first_tf_time = t
            target_time = first_tf_time + int(time_offset_s * 1e9)

        msg = deserialize_message(data, TFMessage)

        if t <= target_time:
            last_msg = msg
            last_time = t
        else:
            if last_msg is not None:
                break
            else:
                # No message before target_time; use this first one.
                last_msg = msg
                last_time = t
                break

    if last_msg is None:
        raise RuntimeError("No /tf messages found in bag.")

    if first_tf_time is not None:
        offset_s = (last_time - first_tf_time) / 1e9
    else:
        offset_s = 0.0

    print(f"[bag_tf_graph] Using /tf snapshot at bag time {last_time} ns "
          f"(offset ~ {offset_s:.6f} s from first /tf).")

    return last_msg


# -------------------------------------------------------------------
# Public class
# -------------------------------------------------------------------

class BagTFGraph:
    """
    TF graph built from a ROS2 bag (static + optional one /tf snapshot).

    Internal representation:
      neighbors[frame] = list of (neighbor_frame, T_frame_to_neighbor)
      children[parent] = list of children (for tree display)

    All transforms are 4x4 numpy arrays mapping **source frame coords → target frame coords**.
    """

    def __init__(self, neighbors, children):
        self._neighbors = neighbors
        self._children = children

    # ---- Construction -------------------------------------------------

    @classmethod
    def from_bag(cls,
                 bag_path: str,
                 tf_static_topic: str = "/tf_static",
                 tf_topic: str = "/tf",
                 tf_time_offset: float | None = None):
        """
        Build a BagTFGraph from a rosbag2 directory.

        Args:
            bag_path: path to rosbag2 directory.
            tf_static_topic: static TF topic (default '/tf_static').
            tf_topic: dynamic TF topic (default '/tf').
            tf_time_offset: if not None, take one /tf snapshot at
                            ~offset seconds after the first /tf message
                            and merge those transforms into the graph.

        Returns:
            BagTFGraph instance.
        """
        bag_path = os.path.abspath(bag_path)
        if not os.path.isdir(bag_path):
            raise RuntimeError(f"Bag directory does not exist: {bag_path}")

        # 1. Static TF
        neighbors, children = _read_static_tf_from_bag(
            bag_path,
            tf_static_topic=tf_static_topic
        )

        # 2. Optional /tf snapshot
        if tf_time_offset is not None:
            tf_msg = _get_tf_message_at_offset(
                bag_path,
                tf_topic=tf_topic,
                time_offset_s=tf_time_offset
            )
            _add_tfmessage_to_graph(tf_msg, neighbors, children)

        return cls(neighbors, children)

    # ---- Query interface ----------------------------------------------

    def lookup_transform(self, source_frame: str, target_frame: str) -> np.ndarray:
        """
        Find transform from source_frame to target_frame using BFS on the graph.

        Returns:
            4x4 matrix T such that:  p_target = T @ p_source
        where p_* are homogeneous coordinates [x, y, z, 1]^T.

        Raises:
            RuntimeError if frames are missing or not connected.
        """
        source = source_frame.lstrip('/')
        target = target_frame.lstrip('/')

        if source == target:
            return np.eye(4)

        if source not in self._neighbors:
            raise RuntimeError(f"Source frame '{source}' not found in TF graph.")
        if target not in self._neighbors:
            raise RuntimeError(f"Target frame '{target}' not found in TF graph.")

        visited = set([source])
        queue = deque([(source, np.eye(4))])  # (current_frame, T_source_to_current)

        while queue:
            frame, T_source_to_current = queue.popleft()
            for nbr, T_frame_to_nbr in self._neighbors.get(frame, []):
                if nbr in visited:
                    continue

                # T_source_to_nbr = T_frame_to_nbr @ T_source_to_current
                T_source_to_nbr = T_frame_to_nbr @ T_source_to_current

                if nbr == target:
                    return T_source_to_nbr

                visited.add(nbr)
                queue.append((nbr, T_source_to_nbr))

        raise RuntimeError(f"No TF path from '{source}' to '{target}'.")

    def print_tree(self):
        """
        Pretty-print the TF tree(s) using the internal children dict.
        """
        children = self._children
        all_children = {c for cs in children.values() for c in cs}
        roots = [f for f in children.keys() if f not in all_children]
        if not roots:
            print("[bag_tf_graph] No clear root frames (graph may be disconnected).")
            roots = list(children.keys())

        def _print_subtree(frame, indent=""):
            print(f"{indent}{frame}")
            for ch in children.get(frame, []):
                _print_subtree(ch, indent + "  ")

        printed = set()
        for root in roots:
            if root in printed:
                continue
            _print_subtree(root)
            printed.update(children[root])


# -------------------------------------------------------------------
# CLI for quick debugging
# -------------------------------------------------------------------

def _main_cli():
    parser = argparse.ArgumentParser(
        description="Inspect TF graph from a rosbag2 (static + optional /tf snapshot)."
    )
    parser.add_argument("--bag", required=True,
                        help="Path to rosbag2 directory.")
    parser.add_argument("--source-frame", required=True,
                        help="Source frame for transform lookup.")
    parser.add_argument("--target-frame", required=True,
                        help="Target frame for transform lookup.")
    parser.add_argument("--tf-static-topic", default="/tf_static",
                        help="Static TF topic name (default: /tf_static).")
    parser.add_argument("--tf-topic", default="/tf",
                        help="Dynamic TF topic name (default: /tf).")
    parser.add_argument("--tf-time-offset", type=float, default=None,
                        help="Optional offset in seconds from first /tf "
                             "message to take a snapshot and merge into graph.")

    args = parser.parse_args()

    graph = BagTFGraph.from_bag(
        args.bag,
        tf_static_topic=args.tf_static_topic,
        tf_topic=args.tf_topic,
        tf_time_offset=args.tf_time_offset
    )

    print("\n=== TF tree ===")
    graph.print_tree()

    print(f"\n=== Transform {args.source_frame} -> {args.target_frame} ===")
    T = graph.lookup_transform(args.source_frame, args.target_frame)
    np.set_printoptions(precision=6, suppress=True)
    #extract quaternion from matrix upper left 3x3
    R_mat = T[:3, :3]
    quat = R.from_matrix(R_mat).as_quat()  # x, y, z, w
    print(f"quat: {quat}")
    print(f"T: {T}")

if __name__ == "__main__":
    _main_cli()
