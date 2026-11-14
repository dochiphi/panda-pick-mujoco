import numpy as np
import mujoco
import eval_pick as ep

for k in [0.2, 0.4, 0.7, 1.0, 1.5, 2.5]:
    ep.apply_light(k)
    mujoco.mj_resetDataKeyframe(ep.model, ep.data, 0)
    ep.data.qpos[ep.cube_adr:ep.cube_adr + 7] = [0.5, 0.1, 0.025, 1, 0, 0, 0]
    ep.data.qpos[0] = -1.5
    mujoco.mj_forward(ep.model, ep.data)
    ep.renderer.update_scene(ep.data, camera="top_cam")
    img = ep.renderer.render()
    r, g, b = (int(v) for v in img[196, 320])
    ok = r > 150 and r - g > 100 and r - b > 100
    print(f"light x{k:<4} cube pixel RGB = ({r:3d}, {g:3d}, {b:3d})  passes red test: {ok}")
