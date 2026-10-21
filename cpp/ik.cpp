#include <mujoco/mujoco.h>
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vec7 = Eigen::Matrix<double, 7, 1>;

struct IKResult {
  int iterations;
  double pos_err, rot_err;
  Vec7 q;
};

IKResult solve_ik(const mjModel* m, int hand, const Eigen::Vector3d& target,
                  const mjtNum* target_quat, const mjtNum* q_start) {
  mjData* d = mj_makeData(m);            // fresh state each solve (same as Python)
  mju_copy(d->qpos, q_start, m->nq);

  const double damping = 0.01;
  std::vector<mjtNum> jacp(3 * m->nv), jacr(3 * m->nv);
  Eigen::Matrix<double, 6, 1> error;

  int i;
  for (i = 0; i < 300; i++) {
    mj_forward(m, d);

    Eigen::Map<const Eigen::Vector3d> pos(d->xpos + 3 * hand);
    error.head<3>() = target - pos;

    mjtNum cur_conj[4], err_quat[4], rot_err[3];
    mju_negQuat(cur_conj, d->xquat + 4 * hand);
    mju_mulQuat(err_quat, target_quat, cur_conj);
    mju_quat2Vel(rot_err, err_quat, 1.0);
    error.tail<3>() = Eigen::Map<Eigen::Vector3d>(rot_err);

    if (error.norm() < 1e-4) break;

    mj_jacBody(m, d, jacp.data(), jacr.data(), hand);
    Eigen::Matrix<double, 6, 7> J;
    J.topRows<3>()    = Eigen::Map<RowMat>(jacp.data(), 3, m->nv).leftCols<7>();
    J.bottomRows<3>() = Eigen::Map<RowMat>(jacr.data(), 3, m->nv).leftCols<7>();

    Eigen::Matrix<double, 6, 6> A = J * J.transpose()
        + damping * damping * Eigen::Matrix<double, 6, 6>::Identity();
    Vec7 dq = J.transpose() * A.ldlt().solve(error);

    for (int j = 0; j < 7; j++) {
      d->qpos[j] = std::clamp(d->qpos[j] + dq[j], m->jnt_range[2 * j], m->jnt_range[2 * j + 1]);
    }
  }

  IKResult r{i, error.head<3>().norm(), error.tail<3>().norm(),
             Eigen::Map<Vec7>(d->qpos)};
  mj_deleteData(d);
  return r;
}

int main() {
  char err[1000] = "";
  mjModel* m = mj_loadXML(MODEL_PATH, nullptr, err, sizeof(err));
  if (!m) { std::printf("Load error: %s\n", err); return 1; }

  // Home pose and "pointing down" orientation
  mjData* d = mj_makeData(m);
  mj_resetDataKeyframe(m, d, 0);
  mj_forward(m, d);
  int hand = mj_name2id(m, mjOBJ_BODY, "hand");
  std::vector<mjtNum> q_home(d->qpos, d->qpos + m->nq);
  mjtNum down_quat[4];
  mju_copy4(down_quat, d->xquat + 4 * hand);

  const Eigen::Vector3d target(0.4, 0.3, 0.4);

  IKResult r = solve_ik(m, hand, target, down_quat, q_home.data());
  std::printf("Iterations: %d, pos error %.6f m, rot error %.6f rad\n",
              r.iterations, r.pos_err, r.rot_err);
  std::printf("Joint angles (rad):");
  for (int j = 0; j < 7; j++) std::printf(" %.3f", r.q[j]);
  std::printf("\n");

  const int N = 1000;
  auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < N; k++) solve_ik(m, hand, target, down_quat, q_home.data());
  auto t1 = std::chrono::steady_clock::now();
  double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
  std::printf("C++ average solve time: %.1f us over %d runs\n", avg_us, N);

  mj_deleteData(d);
  mj_deleteModel(m);
  return 0;
}
