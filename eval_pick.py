import numpy as np
import cv2
import mujoco
from scene import build_model

CAM_POS = np.array([0.5, 0.0, 1.0])
FOVY, H, W = 60, 480, 640
CUBE_TOP_Z = 0.05
HAND_TO_TIP = 0.1034

model = build_model()
data = mujoco.MjData(model)
hand_id = model.body("hand").id
cube_adr = model.jnt_qposadr[model.body("cube").jntadr[0]]
renderer = mujoco.Renderer(model, height=H, width=W)
dt = model.opt.timestep

def detect_cube():
    renderer.update_scene(data, camera="top_cam")
    img = renderer.render()
    r, g, b = (img[:, :, i].astype(int) for i in range(3))
    mask = ((r > 150) & (r - g > 100) & (r - b > 100)).astype(np.uint8) * 255
    m = cv2.moments(mask)
    if m["m00"] == 0:
        return None
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
    return d.qpos[:7].copy()

# Home pose and "pointing down" orientation
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
q_home = data.qpos.copy()
down_quat = data.xquat[hand_id].copy()

def run_trial(cube_xy, yaw=0.0):
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[cube_adr:cube_adr + 3] = [cube_xy[0], cube_xy[1], 0.025]
    data.qpos[cube_adr + 3:cube_adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    data.qpos[0] = -1.5                       # observation pose
    mujoco.mj_forward(model, data)
    q_obs = data.qpos[:7].copy()

    xy = detect_cube()
    if xy is None:
        return False, np.nan, 0.0, np.nan
    det_err = np.hypot(*(xy - cube_xy)) * 1000
    yaw_est = detect_yaw()
    yaw_err = np.degrees((yaw_est - yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)
    grasp_quat = down_quat.copy()
    if ALIGN:
        qz = np.array([np.cos(yaw_est / 2), 0, 0, np.sin(yaw_est / 2)])
        mujoco.mju_mulQuat(grasp_quat, qz, down_quat)

    q_above = solve_ik(np.array([xy[0], xy[1], 0.30]), grasp_quat, q_home)
    seed = q_home.copy(); seed[:7] = q_above
    q_grasp = solve_ik(np.array([xy[0], xy[1], CUBE_TOP_Z / 2 + HAND_TO_TIP]), grasp_quat, seed)

    plan = [(q_above, 255, 3.0), (q_grasp, 255, 2.0),
            (q_grasp, 0, 1.0), (q_above, 0, 2.0), (q_above, 0, 0.5)]
    q_prev = q_obs
    data.ctrl[7] = 255
    for q_tgt, grip, dur in plan:
        n = int(dur / dt)
        for k in range(n):
            s = (k + 1) / n
            s = 3 * s**2 - 2 * s**3
            data.ctrl[:7] = q_prev + s * (q_tgt - q_prev)
            data.ctrl[7] = grip
            data.qfrc_applied[:7] = data.qfrc_bias[:7]
            mujoco.mj_step(model, data)
        q_prev = q_tgt

    z = data.body("cube").xpos[2]
    return z > 0.1, det_err, z, yaw_err

def detect_yaw():
    renderer.update_scene(data, camera="top_cam")
    img = renderer.render()
    r, g, b = (img[:, :, i].astype(int) for i in range(3))
    mask = ((r > 150) & (r - g > 100) & (r - b > 100)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, _, angle = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    a = (angle + 45) % 90 - 45        # square: fold into [-45, 45) deg
    return -np.radians(a)             # image rows point down -> flip sign

rng = np.random.default_rng(0)
import sys
ROTATE = "--rotate" in sys.argv
ALIGN = "--align" in sys.argv
rng_yaw = np.random.default_rng(1)
N = 50
results = []
for t in range(N):
    cube_xy = np.array([rng.uniform(0.35, 0.65), rng.uniform(-0.25, 0.25)])
    yaw = rng_yaw.uniform(-np.pi / 4, np.pi / 4) if ROTATE else 0.0
    ok, det_err, z, yaw_err = run_trial(cube_xy, yaw)
    results.append((ok, det_err, yaw_err, z))
    print(f"Trial {t+1:02d}: yaw {np.degrees(yaw):+5.1f} deg  cube ({cube_xy[0]:.3f}, {cube_xy[1]:.3f})  "
          f"yaw err {yaw_err:+5.2f} deg  detect err {det_err:5.2f} mm  final z {z:.3f}  {'OK' if ok else 'FAIL'}")

oks = [r[0] for r in results]
errs = np.array([r[1] for r in results])
print(f"\nSuccess rate: {sum(oks)}/{N} = {100 * sum(oks) / N:.0f}%")
print(f"Detection error: mean {np.nanmean(errs):.2f} mm, max {np.nanmax(errs):.2f} mm")
yerr = np.abs(np.array([r[2] for r in results]))
print(f"Yaw detection error: mean {np.nanmean(yerr):.2f} deg, max {np.nanmax(yerr):.2f} deg")
zs = np.array([r[3] for r in results if r[0]])
print(f"Final z (successful picks): mean {zs.mean():.3f} m, min {zs.min():.3f} m (ideal 0.197)")
