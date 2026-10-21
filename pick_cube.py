import time
import numpy as np
import cv2
import mujoco
import mujoco.viewer
from scene import build_model

CAM_POS = np.array([0.5, 0.0, 1.0])
FOVY, H, W = 60, 480, 640
CUBE_TOP_Z = 0.05
HAND_TO_TIP = 0.1034   # hand base -> fingertip distance (m)

model = build_model()
data = mujoco.MjData(model)
hand_id = model.body("hand").id

def detect_cube():
    renderer = mujoco.Renderer(model, height=H, width=W)
    renderer.update_scene(data, camera="top_cam")
    img = renderer.render()
    r, g, b = (img[:, :, i].astype(int) for i in range(3))
    mask = ((r > 150) & (r - g > 100) & (r - b > 100)).astype(np.uint8) * 255
    m = cv2.moments(mask)
    if m["m00"] == 0:
        raise RuntimeError("Cube not found in image")
    u, v = m["m10"] / m["m00"], m["m01"] / m["m00"]
    f = (H / 2) / np.tan(np.radians(FOVY / 2))
    Z = CAM_POS[2] - CUBE_TOP_Z
    return np.array([CAM_POS[0] + (u - W / 2) * Z / f,
                     CAM_POS[1] - (v - H / 2) * Z / f])

def solve_ik(target_pos, target_quat, q_start):
    d = mujoco.MjData(model)
    d.qpos[:] = q_start
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    cur_conj, err_quat, rot_err = np.zeros(4), np.zeros(4), np.zeros(3)
    for i in range(300):
        mujoco.mj_forward(model, d)
        pos_err = target_pos - d.xpos[hand_id]
        mujoco.mju_negQuat(cur_conj, d.xquat[hand_id])
        mujoco.mju_mulQuat(err_quat, target_quat, cur_conj)
        mujoco.mju_quat2Vel(rot_err, err_quat, 1.0)
        error = np.concatenate([pos_err, rot_err])
        if np.linalg.norm(error) < 1e-4:
            break
        mujoco.mj_jacBody(model, d, jacp, jacr, hand_id)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])
        dq = J.T @ np.linalg.solve(J @ J.T + 0.01**2 * np.eye(6), error)
        d.qpos[:7] = np.clip(d.qpos[:7] + dq, model.jnt_range[:7, 0], model.jnt_range[:7, 1])
    print(f"  IK -> {np.round(target_pos, 3)}: {i} iters, pos err {np.linalg.norm(pos_err)*1000:.2f} mm")
    return d.qpos[:7].copy()

# --- Setup ---
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
q_home = data.qpos.copy()
down_quat = data.xquat[hand_id].copy()

# --- 1. Observation pose + detect ---
data.qpos[0] = -1.5
mujoco.mj_forward(model, data)
q_obs = data.qpos[:7].copy()
xy = detect_cube()
print(f"Detected cube at x={xy[0]:.4f}, y={xy[1]:.4f}")

# --- 2. Plan with IK ---
above = np.array([xy[0], xy[1], 0.30])
grasp = np.array([xy[0], xy[1], CUBE_TOP_Z / 2 + HAND_TO_TIP])
q_above = solve_ik(above, down_quat, q_home)
seed = q_home.copy(); seed[:7] = q_above
q_grasp = solve_ik(grasp, down_quat, seed)

plan = [("move above", q_above, 255, 3.0),
        ("descend",    q_grasp, 255, 2.0),
        ("close",      q_grasp, 0,   1.0),
        ("lift",       q_above, 0,   2.0)]

# --- 3. Execute ---
data.ctrl[:7], data.ctrl[7] = q_obs, 255
with mujoco.viewer.launch_passive(model, data) as viewer:
    phase, t0, q_prev = 0, data.time, q_obs
    print("Phase:", plan[0][0])
    while viewer.is_running():
        t = data.time
        if phase < len(plan):
            name, q_tgt, grip, dur = plan[phase]
            s = min((t - t0) / dur, 1.0)
            s = 3 * s**2 - 2 * s**3            # smooth start and stop
            data.ctrl[:7] = q_prev + s * (q_tgt - q_prev)
            data.ctrl[7] = grip
            if t - t0 >= dur:
                q_prev, t0, phase = q_tgt, t, phase + 1
                if phase < len(plan):
                    print("Phase:", plan[phase][0])
                else:
                    z = data.body("cube").xpos[2]
                    print(f"Done. Cube height: {z:.3f} m ->", "SUCCESS" if z > 0.1 else "FAILED")
        data.qfrc_applied[:7] = data.qfrc_bias[:7]
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
