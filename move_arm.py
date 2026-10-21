import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")
data = mujoco.MjData(model)

# Start at the "home" pose stored in the model
mujoco.mj_resetDataKeyframe(model, data, 0)
home = data.ctrl.copy()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = data.time

        data.ctrl[:] = home
        data.ctrl[0] = 1.0 * np.sin(0.5 * t)          # base swings +/- 1 rad
        data.ctrl[7] = 255 if np.sin(t) > 0 else 0    # gripper open/close

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
