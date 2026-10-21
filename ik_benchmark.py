import time
import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")
hand_id = model.body("hand").id

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

data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
q_home = data.qpos.copy()
down_quat = data.xquat[hand_id].copy()
target = np.array([0.4, 0.3, 0.4])

q = solve_ik(target, down_quat, q_home)
print("Joint angles (rad):", " ".join(f"{a:.3f}" for a in q))

N = 1000
start = time.perf_counter()
for _ in range(N):
    solve_ik(target, down_quat, q_home)
avg_us = (time.perf_counter() - start) / N * 1e6
print(f"Python average solve time: {avg_us:.1f} us over {N} runs")
