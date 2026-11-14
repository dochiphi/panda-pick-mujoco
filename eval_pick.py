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
from scene import DISTRACTORS
cube2_adr = model.jnt_qposadr[model.body("cube2").jntadr[0]]
dist_adrs = [model.jnt_qposadr[model.body(f"distractor_{n}").jntadr[0]] for n, _ in DISTRACTORS]
renderer = mujoco.Renderer(model, height=H, width=W)
dt = model.opt.timestep

NOISE_SIGMA = 0.0
_rng_noise = np.random.default_rng(2)

def add_noise(img):
    """Gaussian pixel noise (std NOISE_SIGMA, brightness units 0-255)."""
    if NOISE_SIGMA <= 0:
        return img
    noisy = img.astype(np.float32) + _rng_noise.normal(0, NOISE_SIGMA, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)

USE_BLUR = True
USE_LARGEST = True
USE_HSV = True

def _red_mask_raw(img):
    """Pixels that are clearly red. Optional blur first to reduce noise."""
    if USE_BLUR:
        img = cv2.GaussianBlur(img, (5, 5), 0)
    if USE_HSV:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = (hsv[:, :, i].astype(int) for i in range(3))
        return (((h <= 10) | (h >= 170)) & (s > 64) & (v > 50)).astype(np.uint8) * 255
    r, g, b = (img[:, :, i].astype(int) for i in range(3))
    return ((r > 150) & (r - g > 100) & (r - b > 100)).astype(np.uint8) * 255

_light_diffuse0 = model.light_diffuse.copy()
_light_ambient0 = model.light_ambient.copy()
_head_diffuse0 = model.vis.headlight.diffuse.copy()
_head_ambient0 = model.vis.headlight.ambient.copy()

def apply_light(k):
    """Scale all scene lights (and the headlight) by k. k = 1 is the original."""
    model.light_diffuse[:] = _light_diffuse0 * k
    model.light_ambient[:] = _light_ambient0 * k
    model.vis.headlight.diffuse[:] = _head_diffuse0 * k
    model.vis.headlight.ambient[:] = _head_ambient0 * k

TOPFACE_MODE = "v"    # None, "v" (max of RGB) or "l" (average of RGB)

def red_mask(img):
    """Red pixels; with TOPFACE_MODE, keep only the brightest part (the top face)."""
    mask = _red_mask_raw(img)
    if TOPFACE_MODE is None:
        return mask
    if USE_BLUR:
        img = cv2.GaussianBlur(img, (5, 5), 0)
    f = img.astype(float)
    bright = f.max(axis=2) if TOPFACE_MODE == "v" else f.mean(axis=2)
    sel = mask > 0
    if not sel.any():
        return mask
    ref = np.percentile(bright[sel], 95)
    return (sel & (bright >= 0.85 * ref)).astype(np.uint8) * 255

def detect_cube():
    renderer.update_scene(data, camera="top_cam")
    img = add_noise(renderer.render())
    mask = red_mask(img)
    if USE_LARGEST:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        m = cv2.moments(max(contours, key=cv2.contourArea))
    else:
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

USE_CPP = False
_cpp_solver = None

def ik(target_pos, target_quat, q_start):
    """IK through C++ (pybind11) if USE_CPP, else the Python solve_ik."""
    global _cpp_solver
    if not USE_CPP:
        return solve_ik(target_pos, target_quat, q_start)
    if _cpp_solver is None:
        import panda_ik
        _cpp_solver = panda_ik.IKSolver("mujoco_menagerie/franka_emika_panda/scene.xml")
    return _cpp_solver.solve(target_pos, target_quat, q_start[:7]).q

def sample_distractors(rng, cube_xy, n=3, min_dist=0.10):
    """Random distractor spots, at least min_dist from the cube and each other."""
    pts = [np.asarray(cube_xy)]
    while len(pts) < n + 1:
        p = np.array([rng.uniform(0.35, 0.65), rng.uniform(-0.25, 0.25)])
        if all(np.hypot(*(p - q)) >= min_dist for q in pts):
            pts.append(p)
    return pts[1:]

def sample_cube2(rng, cube_xy, min_dist):
    """Second red cube: random spot at least min_dist from the first, random yaw."""
    x, y = sample_distractors(rng, cube_xy, n=1, min_dist=min_dist)[0]
    return (x, y, rng.uniform(-np.pi / 4, np.pi / 4))

LAST_K = None

def sample_cube2_at(rng, cube_xy, yaw, d):
    """Second red cube at exact center distance d, same yaw, side-to-side (gap = d - 0.05)."""
    global LAST_K
    LAST_K = int(rng.integers(4))
    ang = yaw + LAST_K * np.pi / 2
    return (cube_xy[0] + d * np.cos(ang), cube_xy[1] + d * np.sin(ang), yaw)

GRASP_CHECK = True
_d0 = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, _d0, 0)
mujoco.mj_forward(model, _d0)
_ax = _d0.xaxis[model.joint("finger_joint1").id]
FINGER_PHI0 = np.arctan2(_ax[1], _ax[0])   # finger axis angle at grasp yaw 0 (arctan with quadrant check)

def world_to_pixel(x, y, z_top=CUBE_TOP_Z):
    f = (H / 2) / np.tan(np.radians(FOVY / 2))
    Z = CAM_POS[2] - z_top
    return W / 2 + f * (x - CAM_POS[0]) / Z, H / 2 - f * (y - CAM_POS[1]) / Z

def finger_zone_blocked(depth, xy, phi):
    """Occupied sample points in the two finger landing zones (2 x 2 cm, 5 cm from center)."""
    axis = np.array([np.cos(phi), np.sin(phi)])
    perp = np.array([-axis[1], axis[0]])
    count = 0
    for side in (-1, 1):
        center = xy + side * 0.05 * axis
        for a in np.linspace(-0.01, 0.01, 5):
            for b in np.linspace(-0.01, 0.01, 5):
                p = center + a * axis + b * perp
                u, v = world_to_pixel(p[0], p[1])
                ui, vi = int(round(u)), int(round(v))
                if 0 <= vi < H and 0 <= ui < W and depth[vi, ui] < CAM_POS[2] - 0.01:
                    count += 1
    return count

def choose_grasp_yaw(xy, yaw_est):
    """Of the two square-cube grasps (yaw, yaw + 90 deg), pick the one with clearer finger zones."""
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="top_cam")
    depth = renderer.render()
    renderer.disable_depth_rendering()
    options = [yaw_est, yaw_est + np.pi / 2]
    scores = [finger_zone_blocked(depth, xy, y + FINGER_PHI0) for y in options]
    return options[int(np.argmin(scores))], scores

def run_trial(cube_xy, yaw=0.0, on_step=None, distractor_xy=None, cube2=None):
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[cube_adr:cube_adr + 3] = [cube_xy[0], cube_xy[1], 0.025]
    data.qpos[cube_adr + 3:cube_adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    if distractor_xy is not None:
        for adr, (dx, dy) in zip(dist_adrs, distractor_xy):
            data.qpos[adr:adr + 7] = [dx, dy, 0.025, 1, 0, 0, 0]
    if cube2 is not None:
        x2, y2, yaw2 = cube2
        data.qpos[cube2_adr:cube2_adr + 7] = [x2, y2, 0.025, np.cos(yaw2 / 2), 0, 0, np.sin(yaw2 / 2)]
    data.qpos[0] = -1.5                       # observation pose
    mujoco.mj_forward(model, data)
    q_obs = data.qpos[:7].copy()

    xy = detect_cube()
    if xy is None:
        return False, np.nan, 0.0, np.nan
    targets = [(np.asarray(cube_xy), yaw)]
    if cube2 is not None:
        targets.append((np.array(cube2[:2]), cube2[2]))
    dists = [np.hypot(*(xy - t)) for t, _ in targets]
    true_yaw = targets[int(np.argmin(dists))][1]
    det_err = min(dists) * 1000
    yaw_est = detect_yaw()
    yaw_err = np.degrees((yaw_est - true_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)
    if GRASP_CHECK:
        yaw_est, _ = choose_grasp_yaw(xy, yaw_est)
    grasp_quat = down_quat.copy()
    if ALIGN:
        qz = np.array([np.cos(yaw_est / 2), 0, 0, np.sin(yaw_est / 2)])
        mujoco.mju_mulQuat(grasp_quat, qz, down_quat)

    q_above = ik(np.array([xy[0], xy[1], 0.30]), grasp_quat, q_home)
    seed = q_home.copy(); seed[:7] = q_above
    q_grasp = ik(np.array([xy[0], xy[1], CUBE_TOP_Z / 2 + HAND_TO_TIP]), grasp_quat, seed)

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
            if on_step:
                on_step()
        q_prev = q_tgt

    z = max(data.body("cube").xpos[2], data.body("cube2").xpos[2])
    return z > 0.1, det_err, z, yaw_err

def detect_yaw():
    renderer.update_scene(data, camera="top_cam")
    img = add_noise(renderer.render())
    mask = red_mask(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, _, angle = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    a = (angle + 45) % 90 - 45        # square: fold into [-45, 45) deg
    return -np.radians(a)             # image rows point down -> flip sign

ALIGN = False

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    import sys
    ROTATE = "--rotate" in sys.argv
    DISTRACT = "--distract" in sys.argv
    TWO_RED = "--two-red" in sys.argv
    GRASP_CHECK = "--no-grasp-check" not in sys.argv
    MIN2 = float(sys.argv[sys.argv.index("--two-red-min") + 1]) if "--two-red-min" in sys.argv else 0.10
    rng_red2 = np.random.default_rng(4)
    D2 = float(sys.argv[sys.argv.index("--two-red-at") + 1]) if "--two-red-at" in sys.argv else None
    rng_dist = np.random.default_rng(3)
    ALIGN = "--align" in sys.argv
    USE_CPP = "--cpp" in sys.argv
    if "--noise" in sys.argv:
        NOISE_SIGMA = float(sys.argv[sys.argv.index("--noise") + 1])
    USE_BLUR = "--no-blur" not in sys.argv
    USE_LARGEST = "--no-largest" not in sys.argv
    USE_HSV = "--rgb" not in sys.argv
    if "--topface" in sys.argv:
        TOPFACE_MODE = sys.argv[sys.argv.index("--topface") + 1]
        if TOPFACE_MODE == "none":
            TOPFACE_MODE = None
    if "--light" in sys.argv:
        apply_light(float(sys.argv[sys.argv.index("--light") + 1]))
    rng_yaw = np.random.default_rng(1)
    N = 50
    results = []
    for t in range(N):
        cube_xy = np.array([rng.uniform(0.35, 0.65), rng.uniform(-0.25, 0.25)])
        yaw = rng_yaw.uniform(-np.pi / 4, np.pi / 4) if ROTATE else 0.0
        ok, det_err, z, yaw_err = run_trial(cube_xy, yaw, distractor_xy=sample_distractors(rng_dist, cube_xy) if DISTRACT else None, cube2=(sample_cube2(rng_red2, cube_xy, MIN2) if TWO_RED else sample_cube2_at(rng_red2, cube_xy, yaw, D2) if D2 else None))
        results.append((ok, det_err, yaw_err, z))
        print(f"Trial {t+1:02d}: yaw {np.degrees(yaw):+5.1f} deg  cube ({cube_xy[0]:.3f}, {cube_xy[1]:.3f})  "
              f"yaw err {yaw_err:+5.2f} deg  detect err {det_err:5.2f} mm  final z {z:.3f}  {'OK' if ok else 'FAIL'}  dir {LAST_K}")

    oks = [r[0] for r in results]
    errs = np.array([r[1] for r in results])
    print(f"\nSuccess rate: {sum(oks)}/{N} = {100 * sum(oks) / N:.0f}%")
    print(f"Detection error: mean {np.nanmean(errs):.2f} mm, max {np.nanmax(errs):.2f} mm")
    yerr = np.abs(np.array([r[2] for r in results]))
    print(f"Yaw detection error: mean {np.nanmean(yerr):.2f} deg, max {np.nanmax(yerr):.2f} deg")
    zs = np.array([r[3] for r in results if r[0]])
    if zs.size:
        print(f"Final z (successful picks): mean {zs.mean():.3f} m, min {zs.min():.3f} m (ideal 0.197)")
    else:
        print("Final z: no successful picks")
