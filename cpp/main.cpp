#include "ik.h"
#include <chrono>
#include <cstdio>

int main() {
  IKSolver solver(MODEL_PATH);
  const Eigen::Vector3d target(0.4, 0.3, 0.4);
  const Eigen::Vector4d down = solver.home_quat();
  const Vec7 q_home = solver.home_q();

  IKResult r = solver.solve(target, down, q_home);
  std::printf("Iterations: %d, pos error %.6f m, rot error %.6f rad\n",
              r.iterations, r.pos_err, r.rot_err);
  std::printf("Joint angles (rad):");
  for (int j = 0; j < 7; j++) std::printf(" %.3f", r.q[j]);
  std::printf("\n");

  const int N = 1000;
  auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < N; k++) solver.solve(target, down, q_home);
  auto t1 = std::chrono::steady_clock::now();
  double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
  std::printf("C++ average solve time: %.1f us over %d runs\n", avg_us, N);
  return 0;
}
