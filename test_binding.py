import numpy as np
import panda_ik
import eval_pick as ep

solver = panda_ik.IKSolver("mujoco_menagerie/franka_emika_panda/scene.xml")
target = np.array([0.4, 0.3, 0.4])

r = solver.solve(target, solver.home_quat(), solver.home_q())
q_py = ep.solve_ik(target, ep.down_quat, ep.q_home)

print("C++ via Python:", " ".join(f"{a:.3f}" for a in r.q), f"({r.iterations} iters)")
print("Python        :", " ".join(f"{a:.3f}" for a in q_py))
print(f"Max difference: {np.max(np.abs(r.q - q_py)):.2e} rad")
