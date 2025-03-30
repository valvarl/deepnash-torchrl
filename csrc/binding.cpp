#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "core/stratego_env.h"

namespace py = pybind11;

PYBIND11_MODULE(stratego_cpp, m) {
    py::class_<StrategoEnv>(m, "StrategoEnv")
        .def(py::init<std::shared_ptr<StrategoConfig>, const std::string&>(),
             py::arg("config") = nullptr)
        .def("reset", &StrategoEnv::reset, py::arg("seed") = 0)
        .def("step", &StrategoEnv::step)
        .def("generate_observation", &StrategoEnv::generate_observation)
        .def_property_readonly("board", &StrategoEnv::board)
        .def_property_readonly("game_phase", &StrategoEnv::game_phase)
        .def_property_readonly("current_player", &StrategoEnv::current_player);
    
    py::enum_<Player>(m, "Player")
        .value("RED", Player::RED)
        .value("BLUE", Player::BLUE)
        .export_values();

    py::enum_<GamePhase>(m, "GamePhase")
        .value("TERMINAL", GamePhase::TERMINAL)
        .value("DEPLOY", GamePhase::DEPLOY)
        .value("SELECT", GamePhase::SELECT)
        .value("MOVE", GamePhase::MOVE)
        .export_values();
}