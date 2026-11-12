#include "ik.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

IKSolver::IKSolver(const std::string& model_path) {
  char err[1000] = "";
  m_ = mj_loadXML(model_path.c_str(), nullptr, err, sizeof(err));
  if (!m_) throw std::runtime_error(std::string("MuJoCo load error: ") + err);
  hand_ = mj_name2id(m_, mjOBJ_BODY, "hand");
  if (hand_ < 0) throw std::runtime_error("body 'hand' not found");
}

IKSolver::~IKSolver() { mj_deleteModel(m_); }

IKResult IKSolver::solve(const Eigen::Vector3d& target_pos,
                         const Eigen::Vector4d& target_quat,
                         const Vec7& q_start) const {
  mjData* d = mj_makeData(m_);          // fresh state each solve (same as Python)
  if (m_->nkey > 0) mju_copy(d->qpos, m_->key_qpos, m_->nq);  // fingers etc. at home (open)
  for (int j = 0; j < 7; j++) d->qpos[j] = q_start[j];

  mjtNum tq[4] = {target_quat[0], target_quat[1], target_quat[2], target_quat[3]};
  const double damping = 0.01;
  std::vector<mjtNum> jacp(3 * m_->nv), jacr(3 * m_->nv);
  Eigen::Matrix<double, 6, 1> error;

  int i;
  for (i = 0; i < 300; i++) {
    mj_forward(m_, d);

    Eigen::Map<const Eigen::Vector3d> pos(d->xpos + 3 * hand_);
    error.head<3>() = target_pos - pos;

    mjtNum cur_conj[4], err_quat[4], rot_err[3];
    mju_negQuat(cur_conj, d->xquat + 4 * hand_);
    mju_mulQuat(err_quat, tq, cur_conj);
    mju_quat2Vel(rot_err, err_quat, 1.0);
    error.tail<3>() = Eigen::Map<Eigen::Vector3d>(rot_err);

    if (error.norm() < 1e-4) break;

    mj_jacBody(m_, d, jacp.data(), jacr.data(), hand_);
    Eigen::Matrix<double, 6, 7> J;
    J.topRows<3>()    = Eigen::Map<RowMat>(jacp.data(), 3, m_->nv).leftCols<7>();
    J.bottomRows<3>() = Eigen::Map<RowMat>(jacr.data(), 3, m_->nv).leftCols<7>();

    Eigen::Matrix<double, 6, 6> A = J * J.transpose()
        + damping * damping * Eigen::Matrix<double, 6, 6>::Identity();
    Vec7 dq = J.transpose() * A.ldlt().solve(error);

    for (int j = 0; j < 7; j++) {
      d->qpos[j] = std::clamp(d->qpos[j] + dq[j], m_->jnt_range[2 * j], m_->jnt_range[2 * j + 1]);
    }
  }

  IKResult r{i, error.head<3>().norm(), error.tail<3>().norm(), Eigen::Map<Vec7>(d->qpos)};
  mj_deleteData(d);
  return r;
}

Vec7 IKSolver::home_q() const {
  mjData* d = mj_makeData(m_);
  mj_resetDataKeyframe(m_, d, 0);
  Vec7 q = Eigen::Map<Vec7>(d->qpos);
  mj_deleteData(d);
  return q;
}

Eigen::Vector4d IKSolver::home_quat() const {
  mjData* d = mj_makeData(m_);
  mj_resetDataKeyframe(m_, d, 0);
  mj_forward(m_, d);
  Eigen::Vector4d q = Eigen::Map<Eigen::Vector4d>(d->xquat + 4 * hand_);
  mj_deleteData(d);
  return q;
}
