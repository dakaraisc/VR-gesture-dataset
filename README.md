# VR Gesture Dataset

> D. Crowder, R. Zhang, A. E. Block, and W. Yuan, "Requirement-Driven Design of
> Whole-Body Social Tactile Sensing via Virtual Human–Robot Interaction," submitted to
> IROS 2026.

18 participants each performed 8 social touch gestures (`hit`,
`pat`, `poke`, `push`, `rub`, `stroke`, `tap`, `grab`) 20 times on the robot's arm and 20 times on its torso in VR, self-paced with no time limit.

## Data Collection Info

- **Platform:** custom Unity VR environment. Participants wore a Meta Quest 3 headset
(markerless hand tracking via OpenXR, ~30 Hz) and custom haptic gloves (vibrotactile
feedback on contact, six actuators per glove: one per finger + palm).
- **Robot:** Pollen Robotics' Reachy (2021)The torso and arm are represented as convex collision meshes; the hand is represented as a rigid set of capsule colliders. Penetration was allowed
- **Recording:** at every rendering frame, the full 6-DoF pose (position + rotation) of every hand capsule and the robot's mesh were logged, all expressed in the VR scene's world coordinate frame.


**5,520 gesture trials** (2,840 arm + 2,680 torso) across 18 participants, totaling
**738,974 frames**. Every participant has exactly 160 trials (8 gestures × 20
repetitions) in both arm and torso, except `participant_01` and `participant_02`, who
have 140 (the `grab` gesture was not collected for them).

## Data format

Each `participant_XX.parquet` file has one row per (trial, frame), wide-format, and can
be loaded with a single call:

```python
import pandas as pd
df = pd.read_parquet("arm/participant_01/participant_01.parquet")
```

Columns:


| column(s)               | meaning                                                                                |
| ----------------------- | -------------------------------------------------------------------------------------- |
| `trial_id`              | e.g. `"0001_stroke"` — sequential trial index (renumbered, contiguous) + gesture label |
| `trial_ordinal`         | 1-based trial index within this participant; the reliable unique grouping key          |
| `gesture_label`         | one of `hit, pat, poke, push, rub, stroke, tap, grab`                                  |
| `frame`                 | raw source frame number (recording-session-relative, not reset per trial)              |
| `frame_index_in_trial`  | 0-based frame position within the trial (use this for a trial-local time axis)         |
| `{capsule}_pos_x/y/z`   | capsule position (meters), relative to the robot mesh's local frame                    |
| `{capsule}_rot_x/y/z/w` | capsule rotation (quaternion, relative to the robot mesh's local frame)                |


There are 19 hand capsules (`wrist1..wrist4`, `thumb1..thumb3`, `index1..index3`,
`middle1..middle3`, `ring1..ring3`, `pinky1..pinky3`), each with 7 pose columns — see
`assets/hand_capsule_definition.json` for each capsule's physical size (height, radius)
and its local center offset from its tracked joint, which you'll need if you're
reconstructing capsule geometry (rather than just the joint pose) for visualization or
collision analysis.

The per-participant `*_manifest.json` alongside each parquet file lists every trial's
gesture label, frame count, and frame range without needing to load the full parquet.

## Visualizing a trial

`utils/visualize_trial.py` is a self-contained example viewer (Open3D) that plays back
one trial: the robot mesh at identity plus the 19 moving hand capsules.

```bash
python3 visualize_trial.py --body-region arm --participant participant_01 --trial 0001_stroke
```

`--trial` accepts either a `trial_id` (e.g. `0001_stroke`) or a `trial_ordinal` (e.g.
`1`). Controls: Space play/pause, Left/Right arrow step, R reset view, P print camera
params, Q/Esc quit.



The convex hulls used for the torso and arm are located in the assets folder

## Citation

```bibtex
@unpublished{crowder2026requirement,
  title     = {Requirement-Driven Design of Whole-Body Social Tactile Sensing via Virtual Human--Robot Interaction},
  author    = {Crowder, Dakarai and Zhang, Ruohan and Block, Alexis E. and Yuan, Wenzhen},
  note      = {Submitted to IROS 2026},
  year      = {2026}
}
```

