import mujoco
from PIL import Image
from scene import build_model

model = build_model()
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
data.qpos[0] = -1.5   # rotate base so the arm swings out of the camera's view
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera="top_cam")
img = renderer.render()
Image.fromarray(img).save("top_view.png")
print("Saved top_view.png, shape:", img.shape)
