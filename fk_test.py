import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")
data = mujoco.MjData(model)
hand_id = model.body("hand").id

mujoco.mj_resetDataKeyframe(model, data, 0)

for angle in [0.0, 0.5, 1.0]:
    data.qpos[0] = angle          # set base joint angle (rad)
    mujoco.mj_forward(model, data) # compute positions, no physics step
    x, y, z = data.xpos[hand_id]
    print(f"joint1 = {angle:.1f} rad -> hand at x={x:.3f}, y={y:.3f}, z={z:.3f} m")
