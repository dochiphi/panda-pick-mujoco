import numpy as np
import cv2
import mujoco
from scene import build_model

CAM_POS = np.array([0.5, 0.0, 1.0])
FOVY = 60
H, W = 480, 640
CUBE_TOP_Z = 0.05

model = build_model()
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
data.qpos[0] = -1.5   # observation pose
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=H, width=W)
renderer.update_scene(data, camera="top_cam")
img = renderer.render()

# 1. Find red pixels
r, g, b = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
mask = ((r > 150) & (r - g > 100) & (r - b > 100)).astype(np.uint8) * 255
cv2.imwrite("mask.png", mask)

# 2. Center of the red area (pixels)
m = cv2.moments(mask)
u, v = m["m10"] / m["m00"], m["m01"] / m["m00"]

# 3. Pixel -> table position
f = (H / 2) / np.tan(np.radians(FOVY / 2))
Z = CAM_POS[2] - CUBE_TOP_Z
x = CAM_POS[0] + (u - W / 2) * Z / f
y = CAM_POS[1] - (v - H / 2) * Z / f

true = data.body("cube").xpos
print(f"Pixel center:  u={u:.1f}, v={v:.1f}")
print(f"Estimated:     x={x:.4f}, y={y:.4f}")
print(f"True:          x={true[0]:.4f}, y={true[1]:.4f}")
print(f"Error:         {np.hypot(x - true[0], y - true[1]) * 1000:.2f} mm")
