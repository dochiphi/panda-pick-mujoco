# panda-pick-mujoco

Vision-guided pick-and-place with a Franka Panda in MuJoCo. Resume project for robotics/autonomy co-ops.
Repo: https://github.com/dochiphi/panda-pick-mujoco

## Environment
- Windows ARM laptop (Snapdragon) -> WSL2 Ubuntu 24.04, aarch64. No NVIDIA GPU (no Isaac Sim).
- Python 3.12 venv: `source .venv/bin/activate`. Deps in requirements.txt (mujoco==3.16.0).
- `mujoco_menagerie/` is cloned locally and gitignored (Panda model: franka_emika_panda/scene.xml).
- C++: MuJoCo C library at ~/libs/mujoco-3.16.0, Eigen (libeigen3-dev), CMake. Build: `cd cpp && cmake -S . -B build && cmake --build build`.

## Key files
- scene.py: Panda + red 5 cm cube (freejoint) + top_cam (1.0 m above, fovy 60, 640x480).
- eval_pick.py: importable pipeline + 50-trial eval. Flags: --rotate (random yaw ±45°), --align (rotate gripper). run_trial(cube_xy, yaw, on_step) returns (ok, det_err_mm, final_z, yaw_err_deg).
- pick_cube.py: single pick with viewer. record_demo.py: renders demo.mp4 (+ demo.gif via ffmpeg).
- cpp/: IKSolver (ik.h/ik.cpp), test + benchmark exe (main.cpp), pybind11 module panda_ik (bindings.cpp, built into the project root). eval_pick.py --cpp uses it; results identical to Python.

## Pipeline facts
- Observation pose: qpos[0] = -1.5 (arm out of camera view) before detecting.
- Detection: relative red threshold (r>150, r-g>100, r-b>100), centroid, pinhole inverse (f = 415.7 px, Z = 0.95 m, 2.3 mm/pixel). Yaw via cv2.minAreaRect, folded to ±45°, sign flipped (image rows point down).
- IK: damped least squares, lambda 0.01, 6-DoF (position + quaternion error), joint clamping.
- IK must start from the home qpos (fingers open). Closed fingers create contacts in mj_forward: 58 vs 34 µs per solve.
- Control: position actuators + gravity compensation (qfrc_applied[:7] = qfrc_bias[:7]), smoothstep blending.
- Grasp hand height = 0.025 + 0.1034 = 0.128 m. Ideal lifted cube z = 0.197 m. Success = cube z > 0.1 m.

## Current results (baseline to beat or compare against)
- No rotation: 50/50, detection error mean 1.56 mm, max 2.61 mm.
- Rotation, no align: 49/50, worst slip 33 mm. Rotation + align: 50/50, slip 3 mm on every trial. Yaw error mean 0.38°, max 2.57°.
- IK, same process, timer off: C++ 35 µs vs Python 500 µs (14×); pybind11 overhead ≈ 1.6 µs. `import mujoco` sets mjcb_time, making mj_forward 3.5× slower (5.5 → 19.5 µs); compare under equal conditions.

## Robustness (harder tests, done)
- Detector defaults: HSV red test, 5x5 blur, largest red region, top-face filter (V >= 0.85 x 95th pct), depth-based
  grasp selection (two 90-degree options, pick the clear finger zones). Off switches: --rgb, --no-blur,
  --no-largest, --topface none, --no-grasp-check. Test flags: --noise S, --light K, --distract, --two-red,
  --two-red-at D.
- Full results: run_all_tests.py -> results.md (original vs current). Use these numbers in docs, not older runs;
  scene changes shift results ~0.04 mm.
- Known: bright light x2.5 has ~19 mm outliers (cause unknown); 2 mm cube gap 96% (likely region merge).
- Always chain patch && test, and confirm a new flag changes the output before trusting results.

## Rules for this project
- Evaluations use fixed seeds (positions rng seed 0, yaw rng seed 1) so before/after runs are comparable.
- Always measure a baseline before a fix, and report grasp quality (slip = 0.197 - final z), not just success rate.
- Any speed claim: average over many runs, identical work in both versions.
- Commit after each working milestone; keep README results, math, and challenges up to date.
- Never claim results that weren't measured.

## Next tasks
1. DONE: pybind11 module `panda_ik` (cpp/bindings.cpp), used via `eval_pick.py --cpp`, identical eval results.
2. Done: harder tests (see Robustness section above and results.md).
