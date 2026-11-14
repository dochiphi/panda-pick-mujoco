import numpy as np
import cv2
import mujoco
import eval_pick as ep

mujoco.mj_resetDataKeyframe(ep.model, ep.data, 0)
ep.data.qpos[ep.cube_adr:ep.cube_adr + 7] = [0.5, 0.1, 0.025, 1, 0, 0, 0]          # red cube
orange_adr = ep.dist_adrs[2]
ep.data.qpos[orange_adr:orange_adr + 7] = [0.5, -0.1, 0.025, 1, 0, 0, 0]           # orange cube
ep.data.qpos[0] = -1.5
mujoco.mj_forward(ep.model, ep.data)
ep.renderer.update_scene(ep.data, camera="top_cam")
img = ep.renderer.render()

hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# Orange at (0.5, -0.1) appears ~44 px below the image center: row ~284, col 320
for name, (row, col) in [("red", (196, 320)), ("orange", (284, 320))]:
    print(f"{name:6s} center pixel RGB {tuple(int(v) for v in img[row, col])}  HSV {tuple(int(v) for v in hsv[row, col])}")

mask = ep.red_mask(img) > 0
print(f"Red-test pixels in red cube area:    {mask[160:232, 284:356].sum()}")
print(f"Red-test pixels in orange cube area: {mask[248:320, 284:356].sum()}")
