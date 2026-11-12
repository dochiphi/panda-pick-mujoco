#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "ik.h"

namespace py = pybind11;

PYBIND11_MODULE(panda_ik, m) {
  m.doc() = "C++ damped-least-squares IK for the Franka Panda (MuJoCo + Eigen)";

  py::class_<IKResult>(m, "IKResult")
      .def_readonly("iterations", &IKResult::iterations)
      .def_readonly("pos_err", &IKResult::pos_err)
      .def_readonly("rot_err", &IKResult::rot_err)
      .def_readonly("q", &IKResult::q);

  py::class_<IKSolver>(m, "IKSolver")
      .def(py::init<const std::string&>(), py::arg("model_path"))
      .def("solve", &IKSolver::solve,
           py::arg("target_pos"), py::arg("target_quat"), py::arg("q_start"))
      .def("home_q", &IKSolver::home_q)
      .def("home_quat", &IKSolver::home_quat);

  m.def("timer_callback_set", []() { return mjcb_time != nullptr; },
        "True if MuJoCo's profiling timer callback (mjcb_time) is set. "
        "The Python mujoco package sets it on import; it makes mj_forward about 3.5x slower.");
  m.def("disable_timer_callback", []() { mjcb_time = nullptr; },
        "Clear mjcb_time process-wide (disables MuJoCo's internal profiling timers).");
}
