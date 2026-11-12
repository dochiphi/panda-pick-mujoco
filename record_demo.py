import subprocess
import numpy as np
import cv2
import mujoco
import eval_pick as ep

ep.ALIGN = True
W, H = 1280, 720
ep.model.vis.global_.offwidth = W      # allow HD off-screen rendering
ep.model.vis.global_.offheight = H
renderer = mujoco.Renderer(ep.model, height=H, width=W)

cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [0.45, 0.0, 0.2]
cam.distance = 1.6
cam.azimuth = 150
cam.elevation = -25

EVERY = 17                              # grab 1 frame every 17 physics steps
fps = 1 / (EVERY * ep.dt)               # = real-time playback
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
    inset = cv2.resize(ep.renderer.render(), (320, 240))
    frame[20:260, W - 340:W - 20] = inset
    cv2.rectangle(frame, (W - 340, 20), (W - 20, 260), (255, 255, 255), 2)
    cv2.putText(frame, "Top camera", (W - 330, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    ffmpeg.stdin.write(frame.tobytes())

demos = [((0.50, 0.10), 0.0),
         ((0.42, -0.18), np.radians(35)),
         ((0.60, 0.20), np.radians(-40))]

for k, (xy, yaw) in enumerate(demos, 1):
    label = f"Pick {k}/3   cube rotated {np.degrees(yaw):+.0f} deg"
    ok, det_err, z, yaw_err = ep.run_trial(np.array(xy), yaw, on_step=grab)
    print(f"Pick {k}: {'OK' if ok else 'FAIL'}, final z {z:.3f} m")

ffmpeg.stdin.close()
ffmpeg.wait()
print("Saved demo.mp4")
