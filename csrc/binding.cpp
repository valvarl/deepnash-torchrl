#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "core/stratego_env.h"

namespace py = pybind11;

PYBIND11_MODULE(stratego_cpp, m) {
    // py::class_<StrategoEnv>(m, "StrategoEnv")
    //     .def(py::init<std::shared_ptr<StrategoConfig>, const std::string&>(),
    //          py::arg("config") = nullptr)
    //     .def("reset", &StrategoEnv::reset, py::arg("seed") = 0)
    //     .def("step", &StrategoEnv::step)
    //     // .def("generate_observation", &StrategoEnv::generate_observation)
    //     .def_property_readonly("board", &StrategoEnv::board)
    //     .def_property_readonly("game_phase", &StrategoEnv::game_phase)
    //     .def_property_readonly("current_player", &StrategoEnv::current_player);
    
    py::enum_<Piece>(m, "Piece")
        .value("EMPTY", Piece::EMPTY)
        .value("LAKE", Piece::LAKE)
        .value("FLAG", Piece::FLAG)
        .value("SPY", Piece::SPY)
        .value("SCOUT", Piece::SCOUT)
        .value("MINER", Piece::MINER)
        .value("SERGEANT", Piece::SERGEANT)
        .value("LIEUTENANT", Piece::LIEUTENANT)
        .value("CAPTAIN", Piece::CAPTAIN)
        .value("MAJOR", Piece::MAJOR)
        .value("COLONEL", Piece::COLONEL)
        .value("GENERAL", Piece::GENERAL)
        .value("MARSHAL", Piece::MARSHAL)
        .value("BOMB", Piece::BOMB)
        .export_values();

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

    // Position type binding
    py::class_<Pos>(m, "Pos")
        .def(py::init<>())
        .def(py::init<int8_t, int8_t>());
        // .def_readwrite("row", &Pos::row)
        // .def_readwrite("col", &Pos::col)
        // .def("__repr__", [](const Pos& p) {
        //     return "Pos(" + std::to_string(p.row) + ", " + std::to_string(p.col) + ")";
        // });

    // StrategoConfig binding (simplified)
    py::enum_<GameMode>(m, "GameMode")
        .value("ORIGINAL", GameMode::ORIGINAL)
        .value("BARRAGE", GameMode::BARRAGE)
        .export_values();

    py::class_<StrategoConfig, std::shared_ptr<StrategoConfig>>(m, "StrategoConfig")
        .def(py::init<
            size_t,
            size_t,
            const std::unordered_map<Piece, int>&,
            const std::unordered_map<Piece, int>&,
            const std::vector<std::pair<Pos, Pos>>&,
            const std::vector<std::pair<Pos, Pos>>&,
            const std::vector<std::pair<Pos, Pos>>&,
            const std::vector<bool>&,
            const std::vector<bool>&,
            const std::vector<bool>&,
            int,
            int,
            int,
            bool,
            GameMode
        >(),
        py::arg("height"),
        py::arg("width"),
        py::arg("p1_pieces"),
        py::arg("p2_pieces") = std::unordered_map<Piece, int>{},
        py::arg("lakes") = std::vector<std::pair<Pos, Pos>>{},
        py::arg("p1_places_to_deploy") = std::vector<std::pair<Pos, Pos>>{},
        py::arg("p2_places_to_deploy") = std::vector<std::pair<Pos, Pos>>{},
        py::arg("lakes_mask") = std::vector<bool>{},
        py::arg("p1_deploy_mask") = std::vector<bool>{},
        py::arg("p2_deploy_mask") = std::vector<bool>{},
        py::arg("total_moves_limit") = 2000,
        py::arg("moves_since_attack_limit") = 200,
        py::arg("observed_history_entries") = 40,
        py::arg("allow_competitive_deploy") = false,
        py::arg("game_mode") = GameMode::ORIGINAL
        )
        .def_static("from_game_mode", &StrategoConfig::from_game_mode)
        .def_property_readonly("height", &StrategoConfig::height)
        .def_property_readonly("width", &StrategoConfig::width)
        .def_property_readonly("total_moves_limit", &StrategoConfig::total_moves_limit)
        .def_property_readonly("moves_since_attack_limit", &StrategoConfig::moves_since_attack_limit)
        .def_property_readonly("observed_history_entries", &StrategoConfig::observed_history_entries)
        .def_property_readonly("allowed_pieces", &StrategoConfig::allowed_pieces)
        .def_property_readonly("game_mode", &StrategoConfig::game_mode);

    // // Main environment class
    // py::class_<StrategoEnv>(m, "StrategoEnv")
    // .def(py::init<std::shared_ptr<StrategoConfig>, uint32_t>(),
    //      py::arg("config") = nullptr,
    //      py::arg("seed") = 0)
    // .def("reset", &StrategoEnv::reset, py::arg("seed") = 0)
    // .def("step", [](StrategoEnv& env, const Pos& action) {
    //     auto [obs, action_mask, reward, terminated, truncated] = env.step(action);
        
    //     // Convert observation to numpy array
    //     py::array_t<double> obs_array(obs.size(), obs.data());
        
    //     // Convert action mask to numpy array
    //     py::array_t<bool> action_mask_array(action_mask.size(), action_mask.data());
        
    //     return py::make_tuple(
    //         obs_array,
    //         action_mask_array,
    //         reward,
    //         terminated,
    //         truncated
    //     );
    // })
    // .def("generate_observation", [](StrategoEnv& env) {
    //     std::vector<double> obs;
    //     env.generate_observation(obs);
    //     return py::array_t<double>(obs.size(), obs.data());
    // })
    // .def("generate_env_state", [](StrategoEnv& env) {
    //     std::vector<double> obs;
    //     std::vector<bool> action_mask;
    //     env.generate_env_state(obs, action_mask);
        
    //     return py::make_tuple(
    //         py::array_t<double>(obs.size(), obs.data()),
    //         py::array_t<bool>(action_mask.size(), action_mask.data())
    //     );
    // })
    // .def("valid_pieces_to_select", [](StrategoEnv& env, bool is_other_player) {
    //     std::vector<bool> mask;
    //     env.valid_pieces_to_select(mask, is_other_player);
    //     return py::array_t<bool>(mask.size(), mask.data());
    // }, py::arg("is_other_player") = false)
    // .def("valid_destinations", [](StrategoEnv& env) {
    //     std::vector<bool> mask;
    //     env.valid_destinations(mask);
    //     return py::array_t<bool>(mask.size(), mask.data());
    // })
    // .def("get_random_action", &StrategoEnv::get_random_action)
    // .def_property_readonly("board", [](StrategoEnv& env) {
    //     auto& board = env.board();
    //     return py::array_t<int8_t>(
    //         {board.height(), board.width()},
    //         board.data()
    //     );
    // })
    // .def_property_readonly("lakes", [](StrategoEnv& env) {
    //     auto& lakes = env.lakes();
    //     return py::array_t<bool>(
    //         {lakes.height(), lakes.width()},
    //         lakes.data()
    //     );
    // })
    // .def_property_readonly("game_phase", &StrategoEnv::game_phase)
    // .def_property_readonly("current_player", &StrategoEnv::current_player)
    // .def_property_readonly("height", &StrategoEnv::height)
    // .def_property_readonly("width", &StrategoEnv::width);
}