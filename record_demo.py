"""Demo video: 4 picks showing the robustness fixes. Writes demo.mp4."""
import subprocess
import numpy as np
import cv2
import mujoco
import eval_pick as ep

ep.ALIGN = True      # rotate gripper to match the cube (all detector fixes are on by default)
W, H = 1280, 720
ep.model.vis.global_.offwidth = W
ep.model.vis.global_.offheight = H
renderer = mujoco.Renderer(ep.model, height=H, width=W)

cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [0.45, 0.0, 0.2]
cam.distance = 1.6
cam.azimuth = 150
cam.elevation = -25

EVERY = 17
fps = 1 / (EVERY * ep.dt)
ffmpeg = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", f"{fps:.3f}", "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "demo.mp4"],
    stdin=subprocess.PIPE)

step, label = 0, ""

def grab():
    global step
    step += 1
    if step % EVERY:
        return
    renderer.update_scene(ep.data, camera=cam)
    frame = renderer.render().copy()
    ep.renderer.update_scene(ep.data, camera="top_cam")
    inset = cv2.resize(ep.add_noise(ep.renderer.render()), (320, 240))   # what the robot sees
    frame[20:260, W - 340:W - 20] = inset
    cv2.rectangle(frame, (W - 340, 20), (W - 20, 260), (255, 255, 255), 2)
    cv2.putText(frame, "Robot camera", (W - 330, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    ffmpeg.stdin.write(frame.tobytes())

def scene(title, xy, yaw_deg, noise=0.0, light=1.0, distractors=None, cube2=None):
    global label
    ep.NOISE_SIGMA = noise
    ep.apply_light(light)
    label = title
    ok, det_err, z, _ = ep.run_trial(np.array(xy), np.radians(yaw_deg), on_step=grab,
                                     distractor_xy=distractors, cube2=cube2)
    print(f"{title}: {'OK' if ok else 'FAIL'}, detection error {det_err:.2f} mm, final z {z:.3f} m")
    ep.NOISE_SIGMA = 0.0
    ep.apply_light(1.0)

# 2: neighbor exactly 6 cm away along the finger direction (10 mm gap)
yaw2 = np.radians(20)
neighbor = (0.5 + 0.06 * np.cos(yaw2), 0.0 + 0.06 * np.sin(yaw2), yaw2)

scene("1/4  Rotated cube, normal conditions", (0.50, 0.10), 30)
scene("2/4  Second red cube 10 mm away along the fingers", (0.50, 0.00), 20, cube2=neighbor)
scene("3/4  Camera noise + other colored cubes", (0.45, 0.05), -25, noise=40,
      distractors=[(0.60, 0.15), (0.58, -0.12), (0.40, -0.15)])
scene("4/4  Dim light (0.2x)", (0.55, -0.10), 40, light=0.2)

ffmpeg.stdin.close()
ffmpeg.wait()
print("Saved demo.mp4")
