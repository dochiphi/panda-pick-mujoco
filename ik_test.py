import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")
data = mujoco.MjData(model)
hand_id = model.body("hand").id
mujoco.mj_resetDataKeyframe(model, data, 0)

target = np.array([0.4, 0.3, 0.4])   # where we want the hand (m)
damping = 0.01                        # keeps the math stable
jacp = np.zeros((3, model.nv))        # position Jacobian

for i in range(200):
    mujoco.mj_forward(model, data)
    error = target - data.xpos[hand_id]
    if np.linalg.norm(error) < 1e-4:
        break

    mujoco.mj_jacBody(model, data, jacp, None, hand_id)
    J = jacp[:, :7]                   # only the 7 arm joints
    dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), error)

    data.qpos[:7] += dq
    data.qpos[:7] = np.clip(data.qpos[:7], model.jnt_range[:7, 0], model.jnt_range[:7, 1])

print("Iterations:", i)
print("Hand at:", np.round(data.xpos[hand_id], 4))
print("Error (m):", round(np.linalg.norm(error), 6))
print("Joint angles (rad):", np.round(data.qpos[:7], 3))
