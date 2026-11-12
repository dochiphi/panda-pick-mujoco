#pragma once
#include <mujoco/mujoco.h>
#include <Eigen/Dense>
#include <string>

using Vec7 = Eigen::Matrix<double, 7, 1>;

struct IKResult {
  int iterations;
  double pos_err, rot_err;
  Vec7 q;
};

class IKSolver {
 public:
  explicit IKSolver(const std::string& model_path);
  ~IKSolver();
  IKSolver(const IKSolver&) = delete;
  IKSolver& operator=(const IKSolver&) = delete;

  // target_quat is (w, x, y, z); q_start is the 7 arm joint angles
  IKResult solve(const Eigen::Vector3d& target_pos,
                 const Eigen::Vector4d& target_quat,
                 const Vec7& q_start) const;

  Vec7 home_q() const;               // arm joints at the "home" keyframe
  Eigen::Vector4d home_quat() const; // hand orientation at home (pointing down)

 private:
  mjModel* m_ = nullptr;
  int hand_ = -1;
};
