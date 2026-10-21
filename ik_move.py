import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")
hand_id = model.body("hand").id

def solve_ik(target_pos, target_quat, q_start):
    d = mujoco.MjData(model)
    d.qpos[:] = q_start
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    cur_conj = np.zeros(4)
    err_quat = np.zeros(4)
    rot_err = np.zeros(3)

    for i in range(300):
        mujoco.mj_forward(model, d)
        pos_err = target_pos - d.xpos[hand_id]

        # orientation error, as a rotation vector in world frame
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

    print(f"IK: {i} iterations, pos error {np.linalg.norm(pos_err):.6f} m, "
          f"rot error {np.linalg.norm(rot_err):.6f} rad")
    return d.qpos[:7].copy()

# Start at home, remember the "pointing down" orientation
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
down_quat = data.xquat[hand_id].copy()

target = np.array([0.4, 0.3, 0.4])
q_goal = solve_ik(target, down_quat, data.qpos.copy())

with mujoco.viewer.launch_passive(model, data) as viewer:
    data.ctrl[:7] = q_goal
    data.ctrl[7] = 255        # gripper open
    start = time.time()
    while viewer.is_running():
        data.qfrc_applied[:7] = data.qfrc_bias[:7] # cancel gravity (and other bias forces)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
        if time.time() - start > 3 and not hasattr(viewer, "_printed"):
            print("Hand after 3 s:", np.round(data.xpos[hand_id], 4))
            viewer._printed = True
