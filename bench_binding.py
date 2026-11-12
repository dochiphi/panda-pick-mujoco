import time
import numpy as np
import panda_ik
import eval_pick as ep   # imports mujoco, which sets the timer callback

solver = panda_ik.IKSolver("mujoco_menagerie/franka_emika_panda/scene.xml")
target = np.array([0.4, 0.3, 0.4])
down, q7 = solver.home_quat(), solver.home_q()
N = 1000

def bench(fn):
    for _ in range(20):
        fn()
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    return (time.perf_counter() - t0) / N * 1e6

def run(label):
    t_py = bench(lambda: ep.solve_ik(target, ep.down_quat, ep.q_home))
    t_cpp = bench(lambda: solver.solve(target, down, q7))
    print(f"{label:12s} Python IK {t_py:6.1f} us | C++ IK via Python {t_cpp:6.1f} us | {t_py / t_cpp:.1f}x")

print("timer set:", panda_ik.timer_callback_set())
run("Timer ON")
panda_ik.disable_timer_callback()
print("timer set:", panda_ik.timer_callback_set())
run("Timer OFF")
