import mujoco
from scene import build_model

model = build_model()
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)
print("Cube before reset:", data.body("cube").xpos.round(3))

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
print("Cube after reset: ", data.body("cube").xpos.round(3))
print("Hand after reset: ", data.body("hand").xpos.round(3))
