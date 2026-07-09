#!/usr/bin/env python3
"""Example Open3D replay viewer for this dataset: plays back a single
gesture trial (arm/participant_XX/participant_XX.parquet or
torso/participant_XX/participant_XX.parquet).

Every capsule pose in the parquet is already expressed relative to the
robot mesh's own local frame, so the mesh is simply rendered at identity
and never moves -- no per-frame alignment/transform needed to visualize it.

Requires: numpy, pandas, pyarrow (parquet support), open3d.
    pip install numpy pandas pyarrow open3d

Usage (run from anywhere; defaults assume this script lives in
PublicDataset/scripts/):
    python3 visualize_trial.py --body-region arm --participant participant_01 --trial 0001_stroke

Controls: Space play/pause, Left/Right arrow step, R reset view, P print
camera params, Q/Esc quit.
"""
import argparse
import json
import math
import os

import numpy as np
import open3d as o3d
import pandas as pd

# Assumes this script lives in <dataset_root>/scripts/visualize_trial.py.
DEFAULT_OUTPUT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CAPSULE_COLOR = (0.20, 0.55, 0.65)  # neutral teal, not the raw-viewer's red


def quat_to_rotmat_xyzw(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=float)


def make_capsule_mesh(height, radius, orientation, color_rgb, resolution=18):
    """Build a capsule (cylinder capped with two spheres, axis along Y),
    then rotate per `orientation` (0: Y->X, 1: Y stays, 2: Y->Z)."""
    cyl_h = max(height - 2.0 * radius, 0.0)

    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=max(cyl_h, 1e-6), resolution=resolution)
    cyl.compute_vertex_normals()
    R_z_to_y = o3d.geometry.get_rotation_matrix_from_xyz((math.pi / 2, 0, 0))
    cyl.rotate(R_z_to_y, center=(0, 0, 0))

    s0 = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    s1 = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    s0.compute_vertex_normals()
    s1.compute_vertex_normals()
    s0.translate((0, -cyl_h / 2.0, 0))
    s1.translate((0, +cyl_h / 2.0, 0))

    capsule = cyl + s0 + s1
    capsule.compute_vertex_normals()
    capsule.paint_uniform_color(color_rgb)

    direction = int(round(orientation))
    if direction == 0:  # Y -> X
        R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, -math.pi / 2))
        capsule.rotate(R, center=(0, 0, 0))
    elif direction == 2:  # Y -> Z
        R = o3d.geometry.get_rotation_matrix_from_xyz((math.pi / 2, 0, 0))
        capsule.rotate(R, center=(0, 0, 0))

    return capsule


def load_capsule_definitions(path):
    with open(path) as f:
        records = json.load(f)
    return sorted(records, key=lambda r: r["capsule_index"])


def load_trial_frames(parquet_path, trial):
    df = pd.read_parquet(parquet_path)
    trials = df[["trial_ordinal", "trial_id"]].drop_duplicates()
    if trial.isdigit():
        matches = trials[trials["trial_ordinal"] == int(trial)]
    else:
        matches = trials[trials["trial_id"] == trial]
    if matches.empty:
        available = ", ".join(f"{r.trial_ordinal}:{r.trial_id}" for r in trials.itertuples())
        raise SystemExit(f"trial {trial!r} not found. Available trials: {available}")
    ordinal = matches.iloc[0]["trial_ordinal"]
    trial_df = df[df["trial_ordinal"] == ordinal].sort_values("frame_index_in_trial")
    return trial_df, matches.iloc[0]["trial_id"]


class TrialReplayApp:
    def __init__(self, mesh_path, capsule_defs, trial_df, trial_label,
                 capsule_color=DEFAULT_CAPSULE_COLOR, fps=30.0, mesh_scale=1.0,
                 show_axis=False):
        self.trial_df = trial_df.reset_index(drop=True)
        self.capsule_defs = capsule_defs
        self.fps = fps
        self.idx = 0
        self.playing = True

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(f"RoboTouch public trial replay -- {trial_label}", width=1280, height=800)
        self.vis.get_render_option().background_color = np.array([0.92, 0.93, 0.95])

        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        if self.mesh.is_empty():
            raise RuntimeError(f"Failed to load mesh: {mesh_path}")
        self.mesh.compute_vertex_normals()
        self.mesh.scale(mesh_scale, center=(0, 0, 0))
        self.mesh.paint_uniform_color((0.55, 0.55, 0.58))
        self.vis.add_geometry(self.mesh)  # mesh sits at identity for every frame -- never moves

        self.caps = {}
        self.cap_base_v = {}
        self.cap_base_n = {}
        for defn in capsule_defs:
            prefix = defn["column_prefix"]
            cap = make_capsule_mesh(defn["height_m"], defn["radius_m"], defn["orientation"], capsule_color)
            self.caps[prefix] = cap
            self.cap_base_v[prefix] = np.asarray(cap.vertices).copy()
            self.cap_base_n[prefix] = np.asarray(cap.vertex_normals).copy()
            self.vis.add_geometry(cap)

        if show_axis:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
            self.vis.add_geometry(axis)

        self.vis.register_key_callback(ord(" "), self._toggle_play)
        self.vis.register_key_callback(262, self._step_forward)
        self.vis.register_key_callback(263, self._step_back)
        self.vis.register_key_callback(ord("R"), self._reset_view)
        self.vis.register_key_callback(ord("P"), self._print_view)
        self.vis.register_key_callback(ord("Q"), self._quit)
        self.vis.register_key_callback(256, self._quit)

        self._apply_frame(self.idx)
        self._reset_view(None)

    def _toggle_play(self, vis):
        self.playing = not self.playing
        return False

    def _step_forward(self, vis):
        self.playing = False
        self.idx = min(self.idx + 1, len(self.trial_df) - 1)
        self._apply_frame(self.idx)
        return False

    def _step_back(self, vis):
        self.playing = False
        self.idx = max(self.idx - 1, 0)
        self._apply_frame(self.idx)
        return False

    def _reset_view(self, vis):
        vc = self.vis.get_view_control()
        vc.set_lookat([0.0, 0.0, 0.0])
        vc.set_up([0.0, 1.0, 0.0])
        vc.set_front([0.2, -0.1, 1.0])
        vc.set_zoom(0.65)
        return False

    def _quit(self, vis):
        self.vis.close()
        return False

    def _print_view(self, vis):
        vc = self.vis.get_view_control()
        params = vc.convert_to_pinhole_camera_parameters()
        print("\n=== CAMERA PARAMS ===")
        print("intrinsic_K =", np.array2string(params.intrinsic.intrinsic_matrix, precision=8, suppress_small=True))
        print("extrinsic   =", np.array2string(params.extrinsic, precision=8, suppress_small=True))
        print("======================\n")
        return False

    def _apply_rigid(self, mesh, base_v, base_n, R, t):
        mesh.vertices = o3d.utility.Vector3dVector((base_v @ R.T) + t[None, :])
        mesh.vertex_normals = o3d.utility.Vector3dVector(base_n @ R.T)

    def _apply_frame(self, idx):
        row = self.trial_df.iloc[idx]
        for defn in self.capsule_defs:
            prefix = defn["column_prefix"]
            pos = np.array([row[f"{prefix}_pos_x"], row[f"{prefix}_pos_y"], row[f"{prefix}_pos_z"]])
            rot = np.array([row[f"{prefix}_rot_x"], row[f"{prefix}_rot_y"], row[f"{prefix}_rot_z"], row[f"{prefix}_rot_w"]])
            center = np.array([defn["center_offset_m"]["x"], defn["center_offset_m"]["y"], defn["center_offset_m"]["z"]])

            R = quat_to_rotmat_xyzw(rot)
            t = pos + (R @ center)

            cap = self.caps[prefix]
            self._apply_rigid(cap, self.cap_base_v[prefix], self.cap_base_n[prefix], R, t)
            self.vis.update_geometry(cap)

        frame_num = int(row["frame"])
        gesture = row["gesture_label"]
        print(f"\rframe {self.idx + 1}/{len(self.trial_df)} (source frame {frame_num}, gesture={gesture})   ", end="", flush=True)

    def run(self):
        import time
        dt = 1.0 / max(self.fps, 1e-6)
        last = time.time()
        while True:
            if not self.vis.poll_events():
                break
            now = time.time()
            if self.playing and (now - last) >= dt:
                self.idx = (self.idx + 1) % len(self.trial_df)
                self._apply_frame(self.idx)
                last = now
            self.vis.update_renderer()
        self.vis.destroy_window()
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--body-region", required=True, choices=["arm", "torso"])
    ap.add_argument("--participant", required=True, help="e.g. participant_01")
    ap.add_argument("--trial", required=True, help="trial_ordinal (e.g. 1) or trial_id (e.g. 0001_stroke)")
    ap.add_argument("--capsule-color", default=",".join(str(c) for c in DEFAULT_CAPSULE_COLOR), help="RGB 0..1, e.g. '0.2,0.55,0.65'")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--scale", type=float, default=1.25, help="mesh scale factor (matches light.py's validated default)")
    ap.add_argument("--show-axis", action="store_true")
    args = ap.parse_args()

    participant_dir = os.path.join(args.output_root, args.body_region, args.participant)
    parquet_path = os.path.join(participant_dir, f"{args.participant}.parquet")
    if not os.path.exists(parquet_path):
        raise SystemExit(f"not found: {parquet_path}")

    mesh_name = "arm_mesh.obj" if args.body_region == "arm" else "torso_mesh.obj"
    mesh_path = os.path.join(args.output_root, "assets", mesh_name)
    capsule_def_path = os.path.join(args.output_root, "assets", "hand_capsule_definition.json")

    capsule_defs = load_capsule_definitions(capsule_def_path)
    trial_df, trial_id = load_trial_frames(parquet_path, args.trial)
    color = tuple(float(c) for c in args.capsule_color.split(","))

    print(f"Participant: {args.participant} ({args.body_region})")
    print(f"Trial: {trial_id} ({len(trial_df)} frames)")
    print("Controls: Space play/pause | <-/-> step | R reset view | P print camera | Q/Esc quit\n")

    app = TrialReplayApp(
        mesh_path=mesh_path,
        capsule_defs=capsule_defs,
        trial_df=trial_df,
        trial_label=f"{args.participant}/{trial_id}",
        capsule_color=color,
        fps=args.fps,
        mesh_scale=args.scale,
        show_axis=args.show_axis,
    )
    app.run()


if __name__ == "__main__":
    main()
