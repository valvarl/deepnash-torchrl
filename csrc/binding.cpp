#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/stratego_env.h"


namespace py = pybind11;

PYBIND11_MODULE (stratego_cpp, m) {
    py::enum_<Piece> (m, "Piece")
    .value ("EMPTY", Piece::EMPTY)
    .value ("LAKE", Piece::LAKE)
    .value ("FLAG", Piece::FLAG)
    .value ("SPY", Piece::SPY)
    .value ("SCOUT", Piece::SCOUT)
    .value ("MINER", Piece::MINER)
    .value ("SERGEANT", Piece::SERGEANT)
    .value ("LIEUTENANT", Piece::LIEUTENANT)
    .value ("CAPTAIN", Piece::CAPTAIN)
    .value ("MAJOR", Piece::MAJOR)
    .value ("COLONEL", Piece::COLONEL)
    .value ("GENERAL", Piece::GENERAL)
    .value ("MARSHAL", Piece::MARSHAL)
    .value ("BOMB", Piece::BOMB)
    .export_values ();

    py::enum_<Player> (m, "Player")
    .value ("RED", Player::RED)
    .value ("BLUE", Player::BLUE)
    .export_values ();

    py::enum_<GamePhase> (m, "GamePhase")
    .value ("TERMINAL", GamePhase::TERMINAL)
    .value ("DEPLOY", GamePhase::DEPLOY)
    .value ("SELECT", GamePhase::SELECT)
    .value ("MOVE", GamePhase::MOVE)
    .export_values ();

    // StrategoConfig binding (simplified)
    py::enum_<GameMode> (m, "GameMode")
    .value ("CUSTOM", GameMode::CUSTOM)
    .value ("ORIGINAL", GameMode::ORIGINAL)
    .value ("BARRAGE", GameMode::BARRAGE)
    .export_values ();

    py::class_<StrategoConfig, std::shared_ptr<StrategoConfig>> (m, "StrategoConfig")
    .def (
    py::init<size_t, size_t, const std::unordered_map<Piece, int>&, const std::unordered_map<Piece, int>&,
    const std::vector<std::pair<Pos, Pos>>&, const std::vector<std::pair<Pos, Pos>>&,
    const std::vector<std::pair<Pos, Pos>>&, const std::vector<bool>&,
    const std::vector<bool>&, const std::vector<bool>&, int, int, int, bool, GameMode> (),
    py::arg ("height"), py::arg ("width"), py::arg ("p1_pieces"),
    py::arg ("p2_pieces")           = std::unordered_map<Piece, int>{},
    py::arg ("lakes")               = std::vector<std::pair<Pos, Pos>>{},
    py::arg ("p1_places_to_deploy") = std::vector<std::pair<Pos, Pos>>{},
    py::arg ("p2_places_to_deploy") = std::vector<std::pair<Pos, Pos>>{},
    py::arg ("lakes_mask")          = std::vector<bool>{},
    py::arg ("p1_deploy_mask")      = std::vector<bool>{},
    py::arg ("p2_deploy_mask")      = std::vector<bool>{},
    py::arg ("total_moves_limit") = 2000, py::arg ("moves_since_attack_limit") = 200,
    py::arg ("observed_history_entries") = 40,
    py::arg ("allow_competitive_deploy") = false, py::arg ("game_mode") = GameMode::ORIGINAL)
    .def_static ("from_game_mode", &StrategoConfig::from_game_mode)
    .def_property_readonly ("height", &StrategoConfig::height)
    .def_property_readonly ("width", &StrategoConfig::width)
    .def_property_readonly ("total_moves_limit", &StrategoConfig::total_moves_limit)
    .def_property_readonly ("moves_since_attack_limit", &StrategoConfig::moves_since_attack_limit)
    .def_property_readonly ("observed_history_entries", &StrategoConfig::observed_history_entries)
    .def_property_readonly ("allowed_pieces", &StrategoConfig::allowed_pieces)
    .def_property_readonly ("game_mode", &StrategoConfig::game_mode);

    py::class_<MaskedMultiDiscrete> (m, "MaskedMultiDiscrete")
    .def (py::init<const std::vector<int>&, uint32_t> (), py::arg ("nvec"),
    py::arg ("seed") = 0)
    .def ("set_mask", &MaskedMultiDiscrete::set_mask, py::arg ("mask"),
    "Set the binary mask indicating which actions are allowed")
    .def ("sample", &MaskedMultiDiscrete::sample, "Sample a valid action respecting the current mask")
    .def_property_readonly (
    "nvec", [] (const MaskedMultiDiscrete& self) { return self.nvec (); }, "Return the dimension sizes")
    .def_property_readonly ("mask", [] (const MaskedMultiDiscrete& self) {
        auto& mask    = self.mask ();
        auto& nvec    = self.nvec ();
        size_t height = nvec[0], width = nvec[1];
        py::array_t<bool> mask_array ({ height, width });
        auto buf = mask_array.mutable_unchecked<2> ();
        for (size_t i = 0; i < height; ++i) {
            for (size_t j = 0; j < width; ++j) {
                buf (i, j) = mask[i * width + j];
            }
        }
        return mask_array;
    });

    // Main environment class
    py::class_<StrategoEnv, std::shared_ptr<StrategoEnv>> (m, "StrategoEnv")
    .def (py::init<std::shared_ptr<StrategoConfig>, uint32_t> (),
    py::arg ("config") = nullptr, py::arg ("seed") = 0)
    .def ("reset",
    [] (StrategoEnv& env, int seed = 0) {
        auto [obs, action_mask] = env.reset (static_cast<int32_t> (seed));

        // Convert observation to numpy array
        size_t height = env.height (), width = env.width ();
        py::array_t<double> obs_array (
        { obs.size () / height / width, height, width }, obs.data ());

        // Convert action mask to numpy array
        py::array_t<bool> action_mask_array ({ height, width });
        auto buf = action_mask_array.mutable_unchecked<2> ();
        for (size_t i = 0; i < height; ++i) {
            for (size_t j = 0; j < width; ++j) {
                buf (i, j) = action_mask[i * width + j];
            }
        }

        return py::make_tuple (obs_array, action_mask_array);
    })
    .def ("step",
    [] (StrategoEnv& env, const Pos& action) {
        auto [obs, action_mask, reward, terminated, truncated] = env.step (action);

        // Convert observation to numpy array
        size_t height = env.height (), width = env.width ();
        py::array_t<double> obs_array (
        { obs.size () / height / width, height, width }, obs.data ());

        // Convert action mask to numpy array
        py::array_t<bool> action_mask_array ({ height, width });
        auto buf = action_mask_array.mutable_unchecked<2> ();
        for (size_t i = 0; i < height; ++i) {
            for (size_t j = 0; j < width; ++j) {
                buf (i, j) = action_mask[i * width + j];
            }
        }

        return py::make_tuple (obs_array, action_mask_array, reward, terminated, truncated);
    })
    .def ("get_random_action", &StrategoEnv::get_random_action)
    .def ("get_info",
    [] (StrategoEnv& env) {
        py::dict info;

        py::object np_int32 = py::module::import ("numpy").attr ("int32");

        Player current_player = env.current_player ();
        info["cur_player"]    = np_int32 (static_cast<int> (current_player));

        size_t height = env.height (), width = env.width ();
        std::vector<int8_t> board = env.board ();
        if (current_player == Player::BLUE) {
            board = env.rotate_tile<int8_t> (board, true);
        }
        py::array_t<int8_t> board_array ({ height, width }, board.data ());
        info["cur_board"] = board_array;

        py::array_t<int> board_shape_array (2);
        auto board_shape_buf = board_shape_array.mutable_unchecked<1> ();
        board_shape_buf (0)  = static_cast<int> (height);
        board_shape_buf (1)  = static_cast<int> (width);
        info["board_shape"]  = board_shape_array;

        // TODO: think about how much information about allowed_pieces and num_pieces is needed
        std::vector<int> allowed_pieces = env.allowed_pieces ();
        py::array_t<int> allowed_pieces_array (
        allowed_pieces.size (), allowed_pieces.data ());
        info["allowed_pieces"] = allowed_pieces_array;

        info["num_pieces"] = np_int32 (static_cast<int> (allowed_pieces.size ()));
        info["total_moves"]        = np_int32 (env.total_moves ());
        info["moves_since_attack"] = np_int32 (env.moves_since_attack ());

        GamePhase game_phase = env.game_phase ();
        info["game_phase"]   = np_int32 (static_cast<int> (game_phase));

        if (game_phase != GamePhase::MOVE) {
            info["last_selected"] = py::none ();
        } else {
            py::array_t<size_t> last_selected_array (2);
            auto last_selected_buf = last_selected_array.mutable_unchecked<1> ();
            Pos last_selected      = env.last_selected (current_player);
            last_selected_buf (0)  = static_cast<int> (last_selected[0]);
            last_selected_buf (1)  = static_cast<int> (last_selected[1]);
            info["last_selected"]  = last_selected_array;
        }

        return info;
    })
    .def_property_readonly ("action_space",
    [] (StrategoEnv& env) {
        // auto& action_space = env.action_space ();
        // auto& mask         = action_space.mask ();
        // auto& nvec         = action_space.nvec ();
        // size_t height = nvec[0], width = nvec[1];
        // py::array_t<bool> mask_array ({ height, width });
        // auto buf = mask_array.mutable_unchecked<2> ();
        // for (size_t i = 0; i < height; ++i) {
        //     for (size_t j = 0; j < width; ++j) {
        //         buf (i, j) = mask[i * width + j];
        //     }
        // }
        // return py::make_tuple (mask_array, height, width);
        return env.action_space ();
    })
    .def_property_readonly ("board",
    [] (StrategoEnv& env) {
        auto& board = env.board ();
        return py::array_t<int8_t> ({ env.height (), env.width () }, board.data ());
    })
    .def_property_readonly ("lakes",
    [] (StrategoEnv& env) {
        auto& lakes   = env.lakes ();
        size_t height = env.height (), width = env.width ();
        py::array_t<bool> lakes_array ({ height, width });
        auto buf = lakes_array.mutable_unchecked<2> ();
        for (size_t i = 0; i < height; ++i) {
            for (size_t j = 0; j < width; ++j) {
                buf (i, j) = lakes[i * width + j];
            }
        }
        return lakes_array;
    })
    .def_property_readonly ("game_phase", &StrategoEnv::game_phase)
    .def_property_readonly ("current_player", &StrategoEnv::current_player)
    .def_property_readonly ("height", &StrategoEnv::height)
    .def_property_readonly ("width", &StrategoEnv::width)
    .def_property_readonly ("total_moves", &StrategoEnv::total_moves)
    .def_property_readonly ("moves_since_attack", &StrategoEnv::moves_since_attack);
}
