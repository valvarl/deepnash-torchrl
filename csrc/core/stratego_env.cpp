#include "stratego_env.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

#include "prettyprint.hpp"


void roll (std::vector<double>& vec, int shift) {
    if (vec.empty ())
        return;

    shift %= static_cast<int> (vec.size ());
    if (shift < 0) {
        shift += vec.size (); // обработка отрицательного сдвига
    }

    std::rotate (vec.begin (), vec.begin () + (vec.size () - shift), vec.end ());
}

StrategoEnv::StrategoEnv (std::shared_ptr<StrategoConfig> config, uint32_t seed)
: config_ (std::move (config)), game_phase_ (GamePhase::TERMINAL),
  height_ (config_->height ()), width_ (config_->width ()), p1_ (Player::RED),
  p2_ (Player::BLUE), action_space_ ({ 0, 0 }) {
    if (!config_) {
        config_ = std::make_shared<StrategoConfig> (
        StrategoConfig::from_game_mode (GameMode::ORIGINAL));
    }

    action_space_ = MaskedMultiDiscrete (
    { static_cast<int> (height_), static_cast<int> (width_) }, seed);
}

std::tuple<std::vector<double>, std::vector<bool>> StrategoEnv::reset (uint32_t seed) {
    if (height_ != config_->height () || width_ != config_->width ()) {
        height_       = config_->height ();
        width_        = config_->width ();
        action_space_ = MaskedMultiDiscrete (
        { static_cast<int> (height_), static_cast<int> (width_) }, seed);
    }

    game_phase_     = GamePhase::DEPLOY;
    current_player_ = Player::RED;
    board_ = std::vector<int8_t> (height_ * width_, static_cast<int8_t> (Piece::EMPTY)),
    lakes_ = config_->lakes_mask ();

    total_moves_limit_        = config_->total_moves_limit ();
    moves_since_attack_limit_ = config_->moves_since_attack_limit ();
    observed_history_entries_ = config_->observed_history_entries ();

    total_moves_        = 0;
    moves_since_attack_ = 0;

    allowed_pieces_ = config_->allowed_pieces ();

    board_ = std::vector<int8_t> (height_ * width_, static_cast<int8_t> (Piece::EMPTY));
    lakes_ = config_->lakes_mask ();
    for (size_t i = 0; i < height_; ++i) {
        for (size_t j = 0; j < width_; ++j) {
            if (lakes_[i * width_ + j]) {
                board_[i * width_ + j] = static_cast<int8_t> (Piece::LAKE);
            }
        }
    }

    std::vector<int> p1_pieces_vec (static_cast<int> (Piece::MARSHAL) + 1, 0);
    for (const auto& [piece, count] : config_->p1_pieces ()) {
        p1_pieces_vec[static_cast<int> (piece)] = count;
    }

    std::vector<int> p2_pieces_vec (static_cast<int> (Piece::MARSHAL) + 1, 0);
    for (const auto& [piece, count] : config_->p2_pieces ()) {
        p2_pieces_vec[static_cast<int> (piece)] = count;
    }

    p1_.generate_state (p1_pieces_vec, config_->p1_deploy_mask (),
    observed_history_entries_, height_, width_);
    p2_.generate_state (p2_pieces_vec, rotate_tile (config_->p2_deploy_mask_, false),
    observed_history_entries_, height_, width_);

    // Сброс счётчиков
    two_square_detector_.reset ();
    chasing_detector_.reset ();

    std::vector<double> obs;
    std::vector<bool> action_mask;
    generate_env_state (obs, action_mask);
    return { std::move (obs), std::move (action_mask) };
}

void StrategoEnv::generate_observation (std::vector<double>& obs) const {
    obs.reserve (allowed_pieces_.size () * 3 + observed_history_entries_ + 6);

    // 1. Lakes
    for (size_t i = 0; i < height_; ++i) {
        for (size_t j = 0; j < width_; ++j) {
            obs.push_back (lakes_[i * width_ + j] ? 1.0 : 0.0);
        }
    }

    // 2. Private observation
    for (int piece_val : allowed_pieces_) {
        for (size_t i = 0; i < height_; ++i) {
            for (size_t j = 0; j < width_; ++j) {
                int8_t val        = board_[i * width_ + j];
                bool active_piece = val > static_cast<int8_t> (Piece::LAKE);
                obs.push_back ((active_piece && val == piece_val) ? 1.0 : 0.0);
            }
        }
    }

    // 3. Public observation and move history
    if (game_phase_ == GamePhase::DEPLOY) {
        obs.resize (obs.size () +
        (allowed_pieces_.size () * 2 + observed_history_entries_) * height_ * width_,
        0.0);
    } else {
        std::vector<double> public_obs, opp_public_obs;
        const PlayerStateHandler& cur = player_state (current_player_);
        const PlayerStateHandler& opp = player_state (current_player_, true);

        get_public_obs (cur.public_obs_info (), cur.unrevealed (),
        cur.pieces (), cur.movable_pieces (), public_obs);
        get_public_obs (opp.public_obs_info (), opp.unrevealed (),
        opp.pieces (), opp.movable_pieces (), opp_public_obs);
        const std::vector<double>& move_matrix =
        player_state (current_player_).observed_moves ();

        obs.insert (obs.end (), public_obs.begin (), public_obs.end ());
        obs.insert (obs.end (), opp_public_obs.begin (), opp_public_obs.end ());
        obs.insert (obs.end (), move_matrix.begin (), move_matrix.end ());
    }

    // 4. Scalar info
    double total_moves_ratio = static_cast<double> (total_moves_) / total_moves_limit_;
    double since_attack_ratio =
    static_cast<double> (moves_since_attack_) / moves_since_attack_limit_;

    for (size_t i = 0; i < height_; ++i) {
        for (size_t j = 0; j < width_; ++j) {
            obs.push_back (total_moves_ratio);
            obs.push_back (since_attack_ratio);
            obs.push_back (game_phase_ == GamePhase::DEPLOY ? 1.0 : 0.0);
            obs.push_back (game_phase_ == GamePhase::MOVE ? 1.0 : 0.0);
        }
    }

    // 5. last_selected
    const Pos& last_selected = player_state (current_player_).last_selected ();
    for (size_t i = 0; i < height_; ++i) {
        for (size_t j = 0; j < width_; ++j) {
            obs.push_back ((i == last_selected[0] && j == last_selected[1]) ? 1.0 : 0.0);
        }
    }
}

void StrategoEnv::generate_env_state (std::vector<double>& obs, std::vector<bool>& action_mask) {
    generate_observation (obs);
    if (game_phase_ == GamePhase::DEPLOY) {
        valid_spots_to_place (action_mask);
    } else if (game_phase_ == GamePhase::SELECT) {
        valid_pieces_to_select (action_mask);
    } else {
        valid_destinations (action_mask);
    }
    action_space_.set_mask (action_mask);
    std::cout << "action_mask: " << action_space_.mask () << "\n";
}

void StrategoEnv::get_public_obs (const std::array<std::vector<bool>, 3>& public_obs_info,
const std::vector<int>& unrevealed,
const std::vector<int>& pieces,
const std::vector<int>& movable_pieces,
std::vector<double>& public_obs) const {
    size_t num_pieces = pieces.size ();
    public_obs.reserve (num_pieces * height_ * width_);

    // Суммы для нормализации
    int sum_all = 0;
    for (int p : pieces)
        sum_all += unrevealed[p];

    int sum_movable = 0;
    for (int p : movable_pieces)
        sum_movable += unrevealed[p];

    // Расчёт вероятностей
    std::vector<double> probs_unmoved (num_pieces, 0.0);
    std::vector<double> probs_moved (num_pieces, 0.0);

    for (size_t i = 0; i < num_pieces; ++i) {
        int piece = pieces[i];
        if (sum_all > 0)
            probs_unmoved[i] = static_cast<double> (unrevealed[piece]) / sum_all;

        if (sum_movable > 0 &&
        std::find (movable_pieces.begin (), movable_pieces.end (), piece) !=
        movable_pieces.end ()) {
            probs_moved[i] = static_cast<double> (unrevealed[piece]) / sum_movable;
        }
    }

    for (size_t i = 0; i < num_pieces; ++i) {
        for (size_t r = 0; r < height_; ++r) {
            for (size_t c = 0; c < width_; ++c) {
                double val_unmoved =
                public_obs_info[0][r * width_ + c] ? probs_unmoved[i] : 0.0;
                double val_moved =
                public_obs_info[1][r * width_ + c] ? probs_moved[i] : 0.0;
                double val_revealed = public_obs_info[2][r * width_ + c] &&
                static_cast<int> (pieces[i]) ==
                static_cast<int> (public_obs_info[2][r * width_ + c]) ?
                1.0 :
                0.0;
                public_obs.push_back (val_unmoved + val_moved + val_revealed);
            }
        }
    }
}

void StrategoEnv::encode_move (const Pos& src, const Pos& dest, std::vector<double>& encoding) const {
    encoding.resize (height_ * width_, 0.0);
    int8_t src_piece  = board_[src[0] * width_ + src[1]];
    int8_t dest_piece = board_[dest[0] * width_ + dest[1]];

    if (dest_piece == static_cast<int8_t> (Piece::EMPTY)) {
        encoding[dest[0] * width_ + dest[1]] = 1.0;
        encoding[src[0] * width_ + src[1]]   = -1.0;
    } else {
        double weight = 2.0 + (static_cast<double> (src_piece) - 3.0) / 12.0;
        encoding[dest[0] * width_ + dest[1]] = 1.0;
        encoding[src[0] * width_ + src[1]]   = -weight;
    }
}

std::tuple<std::vector<double>, std::vector<bool>, int, bool, bool>
StrategoEnv::step (const Pos& action) {
    std::pair<bool, std::string> action_valid = validate_coord (action);
    if (!action_valid.first) {
        throw std::invalid_argument (action_valid.second);
    }

    int reward      = 0;
    bool terminated = false;
    bool truncated  = false;

    std::vector<double> obs;
    std::vector<bool> action_mask;

    switch (game_phase_) {
    case GamePhase::DEPLOY: {
        valid_spots_to_place (action_mask);
        if (!action_mask[action[0] * width_ + action[1]]) {
            // action = action_space_.sample();
            throw std::invalid_argument ("Invalid deployment location");
        }

        auto& curr_player = (current_player_ == Player::RED) ? p1_ : p2_;
        auto& opp_player  = (current_player_ == Player::RED) ? p2_ : p1_;

        size_t cumulative_count = 0;
        size_t deploy_piece     = 0;

        for (size_t i = 0; i < allowed_pieces_.size (); ++i) {
            size_t piece = allowed_pieces_[i];
            size_t count = curr_player.unrevealed_[piece];

            if (curr_player.deploy_idx_ < cumulative_count + count) {
                deploy_piece = piece;
                break;
            }
            cumulative_count += count;
        }

        board_[action[0] * width_ + action[1]] = deploy_piece;
        ++curr_player.deploy_idx_;

        std::ostringstream oss;

        // oss << "urevealed: " << curr_player.unrevealed_ << std::endl;
        // oss << "allowed_pieces: " << allowed_pieces_ << std::endl;

        oss << "board\n";
        for (size_t y = 0; y < height_; ++y) {
            for (size_t x = 0; x < width_; ++x) {
                oss << std::to_string (board_[y * width_ + x]) << "\t";
            }
            oss << "\n";
        }

        oss << "deploy_piece: " << deploy_piece;
        oss << "\naction[0]: " << std::to_string (action[0])
            << "\naction[1]: " << std::to_string (action[1]);
        oss << "\nindex: " << std::to_string (action[0] * width_ + action[1]);

        std::cout << oss.str () << std::endl;

        bool curr_finish_deploy = curr_player.deploy_idx_ ==
        std::accumulate (
        curr_player.unrevealed_.begin (), curr_player.unrevealed_.end (), 0);
        bool opp_finish_deploy = opp_player.deploy_idx_ ==
        std::accumulate (
        opp_player.unrevealed_.begin (), opp_player.unrevealed_.end (), 0);
        if (opp_finish_deploy && !curr_finish_deploy) {
            break;
        } else if (curr_finish_deploy && opp_finish_deploy) {
            std::vector<bool> action_mask;
            valid_pieces_to_select (action_mask, true);
            // std::cout << action_mask << std::endl;
            bool opp_has_moves = std::any_of (
            action_mask.begin (), action_mask.end (), [] (bool x) { return x; });
            // std::cout << opp_no_moves << std::endl;
            if (!opp_has_moves) {
                valid_pieces_to_select (action_mask, false);
                bool draw_game = std::any_of (action_mask.begin (),
                action_mask.end (), [] (bool x) { return x; });
                reward         = static_cast<int> (draw_game);
                game_phase_    = GamePhase::TERMINAL;
                terminated     = true;
                break;
            }
            game_phase_ = GamePhase::SELECT;
            for (size_t r = 0; r < height_; ++r) {
                for (size_t c = 0; c < width_; ++c) {
                    if (board_[r * width_ + c] > static_cast<int8_t> (Piece::LAKE) &&
                    curr_player.deploy_mask_[r * width_ + c]) {
                        curr_player.public_obs_info_[0][r * width_ + c] = true;
                    }
                    size_t opp_r = height_ - r - 1, opp_c = width_ - c - 1;
                    if (board_[opp_r * width_ + opp_c] < -static_cast<int8_t> (Piece::LAKE) &&
                    opp_player.deploy_mask_[opp_r * width_ + opp_c]) {
                        opp_player.public_obs_info_[0][r * width_ + c] = true;
                    }
                }
            }
        }
        rotate_tile_inplace<int8_t> (board_);
        switch_current_player ();
        break;
    }
    case GamePhase::SELECT: {
        valid_pieces_to_select (action_mask);
        if (!action_mask[action[0] * width_ + action[1]]) {
            // action = action_space_.sample();
            throw std::invalid_argument ("Invalid piece selection");
        }

        auto& player = (current_player_ == Player::RED) ? p1_ : p2_;
        player.set_last_selected (action);
        player.set_last_selected_piece (
        static_cast<Piece> (board_[action[0] * width_ + action[1]]));
        game_phase_ = GamePhase::MOVE;
        break;
    }
    case GamePhase::MOVE: {
        auto src = (current_player_ == Player::RED) ? p1_.last_selected () :
                                                      p2_.last_selected ();
        auto dest = action;

        auto [valid, msg] = check_action_valid (src, dest);
        if (!valid) {
            // action = action_space_.sample();
            // dest = action;
            throw std::invalid_argument (msg);
        }

        auto& curr_player = (current_player_ == Player::RED) ? p1_ : p2_;
        auto& opp_player  = (current_player_ == Player::RED) ? p2_ : p1_;

        curr_player.last_selected_ = action;

        // std::cout << "SRC: " << static_cast<int>(src[0]) << " " << static_cast<int>(src[1]) << std::endl;
        // std::cout << "DEST: " << static_cast<int>(dest[0]) << " " << static_cast<int>(dest[1]) << std::endl;

        int8_t src_piece_val  = board_[src[0] * width_ + src[1]];
        int8_t dest_piece_val = board_[dest[0] * width_ + dest[1]];

        // Check if draw conditions are met
        if (total_moves_ >= total_moves_limit_ || moves_since_attack_ >= moves_since_attack_limit_) {
            terminated = true;
            rotate_tile_inplace<int8_t> (board_);
            switch_current_player ();
            game_phase_ = GamePhase::TERMINAL;
            break;
        }

        // Update Draw conditions
        total_moves_++;
        moves_since_attack_ = (dest_piece_val == static_cast<int8_t> (Piece::EMPTY)) ?
        moves_since_attack_ + 1 :
        0;

        // Update Move Histories
        roll (curr_player.observed_moves_, height_ * width_);
        roll (opp_player.observed_moves_, height_ * width_);
        std::vector<double> move;
        encode_move (src, dest, move);
        std::copy (move.begin (), move.end (), curr_player.observed_moves_.begin ());
        rotate_tile_inplace<double> (move);
        std::copy (move.begin (), move.end (), opp_player.observed_moves_.begin ());

        two_square_detector_.update (
        current_player_, static_cast<Piece> (src_piece_val), src, dest);
        Pos src_rot  = rotate_coord (src);
        Pos dest_rot = rotate_coord (dest);

        auto _src = src, _dest = dest;
        if (current_player_ == Player::BLUE) {
            _src  = src_rot;
            _dest = dest_rot;
        }
        chasing_detector_.update (current_player_,
        static_cast<Piece> (src_piece_val), _src, _dest, board_, height_, width_);

        std::cout << "WARNING: check dest_piece_val negative\n";

        if (src_piece_val == -dest_piece_val) { // Equal Strength
            // remove both pieces
            board_[src[0] * width_ + src[1]]   = 0;
            board_[dest[0] * width_ + dest[1]] = 0;
            for (int k = 0; k < 3; ++k) {
                curr_player.public_obs_info_[k][src[0] * width_ + src[1]]   = 0;
                curr_player.public_obs_info_[k][dest[0] * width_ + dest[1]] = 0;
                opp_player.public_obs_info_[k][src_rot[0] * width_ + src_rot[1]] = 0;
                opp_player.public_obs_info_[k][dest_rot[0] * width_ + dest_rot[1]] = 0;
            }
            curr_player.unrevealed_[src_piece_val]--;
            opp_player.unrevealed_[-dest_piece_val]--;
        } else if ((src_piece_val == static_cast<int8_t> (Piece::SPY) &&
                   dest_piece_val == -static_cast<int8_t> (Piece::MARSHAL)) || // Spy vs Marshal
        (src_piece_val > -dest_piece_val &&
        (src_piece_val == static_cast<int8_t> (Piece::MINER) &&
        dest_piece_val == -static_cast<int8_t> (Piece::BOMB) ||
        dest_piece_val != -static_cast<int8_t> (Piece::BOMB)))) { // attacker is stronger (+Bomb case)
            // remove enemy piece
            board_[src[0] * width_ + src[1]]   = 0;
            board_[dest[0] * width_ + dest[1]] = src_piece_val;
            for (int k = 0; k < 3; ++k) {
                curr_player.public_obs_info_[k][src[0] * width_ + src[1]] = 0;
            }
            if (dest_piece_val != static_cast<int8_t> (Piece::EMPTY)) {
                curr_player.public_obs_info_[2][dest[0] * width_ + dest[1]] = src_piece_val;
                for (int k = 0; k < 3; ++k) {
                    opp_player.public_obs_info_[k][dest_rot[0] * width_ + dest_rot[1]] = 0;
                }
                curr_player.unrevealed_[src_piece_val]--;
                opp_player.unrevealed_[-dest_piece_val]--;
            } else {
                bool scout_move = (src[0] == dest[0] && abs (src[1] - dest[1]) > 1) ||
                (src[1] == dest[1] && abs (src[0] - dest[0]) > 1);
                if (scout_move) {
                    curr_player.public_obs_info_[2][dest[0] * width_ + dest[1]] = src_piece_val;
                    curr_player.unrevealed_[src_piece_val]--;
                } else {
                    curr_player.public_obs_info_[1][dest[0] * width_ + dest[1]] = 1;
                }
            }
            if (dest_piece_val == -static_cast<int8_t> (Piece::FLAG)) {
                reward     = 1;
                terminated = true;
            }
        } else if (src_piece_val < -dest_piece_val ||
        dest_piece_val == -static_cast<int8_t> (Piece::BOMB)) {
            // remove attacker
            board_[src[0] * width_ + src[1]] = 0;
            for (int k = 0; k < 3; ++k) {
                curr_player.public_obs_info_[k][src[0] * width_ + src[1]] = 0;
            }
            for (int k = 0; k < 2; ++k) {
                opp_player.public_obs_info_[k][dest_rot[0] * width_ + dest_rot[1]] = 0;
            }
            opp_player.public_obs_info_[2][dest_rot[0] * width_ + dest_rot[1]] = -dest_piece_val;
            curr_player.unrevealed_[src_piece_val]--;
            opp_player.unrevealed_[-dest_piece_val]--;
        } else {
            throw std::runtime_error ("Move was left unprocessed");
        }

        rotate_tile_inplace<int8_t> (board_);
        switch_current_player ();

        // Check if any pieces can be moved. If one player has no movable pieces, the other player wins.
        // If both players have no movable pieces, the game is a draw.
        if (!terminated) {
            std::vector<bool> action_mask;
            valid_pieces_to_select (action_mask, false);
            bool curr_has_moves = std::any_of (
            action_mask.begin (), action_mask.end (), [] (bool x) { return x; });
            if (!curr_has_moves) {
                action_mask.clear ();
                valid_pieces_to_select (action_mask, true);
                reward = static_cast<int> (std::any_of (action_mask.begin (),
                action_mask.end (), [] (bool x) { return x; }));
                game_phase_ = GamePhase::TERMINAL;
                terminated  = true;
                break;
            }
        }
        game_phase_ = terminated ? GamePhase::TERMINAL : GamePhase::SELECT;
        break;
    }

    case GamePhase::TERMINAL:
        throw std::runtime_error ("Game has already terminated");
    }

    generate_env_state (obs, action_mask);
    return { std::move (obs), std::move (action_mask), reward, terminated, truncated };
}

inline std::pair<bool, std::string> StrategoEnv::validate_coord (const Pos& coord) const {
    if (coord[0] < 0 || coord[0] >= height_) {
        return { false, "Source row is out of bounds" };
    } else if (coord[1] < 0 || coord[1] >= width_) {
        return { false, "Source column is out of bounds" };
    }
    return { true, "" };
}

std::pair<bool, std::string>
StrategoEnv::check_action_valid (const Pos& src, const Pos& dest) const {
    auto valid = validate_coord (src);
    if (!valid.first) {
        return valid;
    }

    valid = validate_coord (dest);
    if (!valid.first) {
        return valid;
    }

    int8_t src_piece_val  = board_[src[0] * width_ + src[1]];
    int8_t dest_piece_val = board_[dest[0] * width_ + dest[1]];

    std::cout << static_cast<int> (src_piece_val) << " "
              << static_cast<int> (dest_piece_val);
    std::cout << "SRC DEST: " << static_cast<int> (src[0]) << " "
              << static_cast<int> (src[1]) << " " << static_cast<int> (dest[0])
              << " " << static_cast<int> (dest[1]) << "\n";

    if (src_piece_val < static_cast<int8_t> (Piece::SPY)) {
        return { false, "Selected piece cannot be moved by player" };
    }

    if (abs (dest_piece_val) == static_cast<int8_t> (Piece::LAKE)) {
        return { false, "Destination is an obstacle" };
    }

    if (dest_piece_val > static_cast<int8_t> (Piece::LAKE)) {
        return { false, "Destination is already occupied by player's piece" };
    }

    if (src_piece_val != static_cast<int8_t> (Piece::SCOUT)) {
        if (!(src[0] == dest[0] && abs (src[1] - dest[1]) == 1) &&
        !(src[1] == dest[1] && abs (src[0] - dest[0]) == 1)) {
            return { false, "Invalid move" };
        }
    } else {
        if (src[0] != dest[0] && src[1] != dest[1]) {
            return { false, "Scouts can only move in straight lines" };
        }
        if (src[0] == dest[0]) {
            for (auto i = std::min (src[1], dest[1]) + 1;
                 i < std::max (src[1], dest[1]); ++i) {
                if (board_[src[0] * width_ + i] != 0) {
                    return { false, "Pieces in the path of scout" };
                }
            }
        } else if (src[1] == dest[1]) {
            for (auto i = std::min (src[0], dest[0]) + 1;
                 i < std::max (src[0], dest[0]); ++i) {
                if (board_[i * width_ + src[1]] != 0) {
                    return { false, "Pieces in the path of scout" };
                }
            }
        } else {
            return { false, "Invalid move" };
        }
    }

    if (!two_square_detector_.validate_move (
        current_player_, Piece (src_piece_val), src, dest)) {
        return { false, "Two-square rule violation" };
    }

    auto src_ = src, dest_ = dest;
    if (current_player_ == Player::BLUE) {
        src_  = rotate_coord (src);
        dest_ = rotate_coord (dest);
    }
    if (!chasing_detector_.validate_move (current_player_,
        Piece (src_piece_val), src_, dest_, board_, height_, width_)) {
        return { false, "More-square rule violation" };
    }

    return { true, "Valid Action" };
}

void StrategoEnv::valid_spots_to_place (std::vector<bool>& action_mask) const {
    action_mask.clear ();
    action_mask.resize (height_ * width_, false);

    const auto& deploy_mask = player_state (current_player_).deploy_mask ();

    for (size_t y = 0; y < height_; ++y) {
        for (size_t x = 0; x < width_; ++x) {
            if (board_[y * width_ + x] == static_cast<int8_t> (Piece::EMPTY) &&
            deploy_mask[y * width_ + x]) {
                action_mask[y * width_ + x] = true;
            }
        }
    }
}

void StrategoEnv::valid_pieces_to_select (std::vector<bool>& action_mask,
bool is_other_player) const {
    action_mask.clear ();
    action_mask.resize (height_ * width_, false);

    // Get the appropriate board view
    std::vector<int8_t> board_view = board_;
    if (is_other_player) {
        board_view = rotate_tile<int8_t> (board_, true);
    }

    // Create padded board with lakes around the edges
    std::vector<int8_t> padded_board (
    (height_ + 2) * (width_ + 2), static_cast<int8_t> (Piece::LAKE));
    for (size_t i = 0; i < height_; ++i) {
        for (size_t j = 0; j < width_; ++j) {
            padded_board[(i + 1) * width_ + j + 1] = board_view[i * width_ + j];
            if (padded_board[(i + 1) * width_ + j + 1] == -static_cast<int8_t> (Piece::LAKE)) {
                padded_board[(i + 1) * width_ + j + 1] =
                static_cast<int8_t> (Piece::LAKE);
            }
        }
    }

    // Check for surrounded pieces
    std::vector<bool> surrounded (height_ * width_, false);
    for (size_t i = 0; i < height_; ++i) {
        for (size_t j = 0; j < width_; ++j) {
            bool left = padded_board[(i + 1) * width_ + j] >=
            static_cast<int8_t> (Piece::LAKE);
            bool right = padded_board[(i + 1) * width_ + j + 2] >=
            static_cast<int8_t> (Piece::LAKE);
            bool up =
            padded_board[i * width_ + j + 1] >= static_cast<int8_t> (Piece::LAKE);
            bool down = padded_board[(i + 2) * width_ + j + 1] >=
            static_cast<int8_t> (Piece::LAKE);

            surrounded[i * width_ + j] = left && right && up && down;
        }
    }

    // Handle two-square and chasing rules if there's a last selected piece
    Player player = is_other_player ?
    (current_player_ == Player::RED ? Player::BLUE : Player::RED) :
    current_player_;

    const PlayerStateHandler& p = (player == Player::RED) ? p1_ : p2_;
    Pos last_pos                = p.last_selected ();
    Piece last_piece            = p.last_selected_piece ();

    if (last_pos[0] != -1 && last_pos[1] != -1) {
        // Check two-square rule
        auto two_square_valid =
        two_square_detector_.validate_select (player, last_piece, last_pos);

        // Check chasing rule
        Pos rotated_last_pos = last_pos;
        if (player == Player::BLUE) {
            rotated_last_pos = rotate_coord (last_pos);
        }
        auto chasing_valid = chasing_detector_.validate_select (
        player, last_piece, rotated_last_pos, board_view, height_, width_);

        if (!two_square_valid.first || !chasing_valid.first) {
            std::vector<bool> restricted_mask (height_ * width_, false);

            // Apply two-square restrictions
            if (!two_square_valid.first) {
                Pos start_pos = two_square_valid.second.first;
                Pos end_pos   = two_square_valid.second.second;

                if (start_pos == end_pos) {
                    restricted_mask[start_pos[0] * width_ + start_pos[1]] = true;
                } else {
                    if (start_pos[0] == end_pos[0]) {
                        int min_col = std::min (start_pos[1], end_pos[1]);
                        int max_col = std::max (start_pos[1], end_pos[1]);
                        for (int j = min_col; j <= max_col; ++j) {
                            restricted_mask[start_pos[0] * width_ + j] = true;
                        }
                    } else {
                        int min_row = std::min (start_pos[0], end_pos[0]);
                        int max_row = std::max (start_pos[0], end_pos[0]);
                        for (int i = min_row; i <= max_row; ++i) {
                            restricted_mask[i * width_ + start_pos[1]] = true;
                        }
                    }
                }
            }

            // Apply chasing restrictions
            if (!chasing_valid.first) {
                for (const Pos& pos : chasing_valid.second) {
                    Pos actual_pos = pos;
                    if (player == Player::BLUE) {
                        actual_pos = rotate_coord (pos);
                    }
                    restricted_mask[actual_pos[0] * width_ + actual_pos[1]] = true;
                }
            }

            // Check if piece is completely surrounded considering restrictions
            int surrounded_count                                = 0;
            const std::array<std::pair<int, int>, 4> directions = { { { -1, 0 },
            { 1, 0 }, { 0, -1 }, { 0, 1 } } };

            for (const auto& dir : directions) {
                Pos check_pos = { static_cast<int8_t> (last_pos[0] + dir.first),
                    static_cast<int8_t> (last_pos[1] + dir.second) };

                while (check_pos[0] >= 0 && check_pos[0] < height_ &&
                check_pos[1] >= 0 && check_pos[1] < width_) {
                    if (!restricted_mask[check_pos[0] * width_ + check_pos[1]]) {
                        int8_t piece_val =
                        board_view[check_pos[0] * width_ + check_pos[1]];
                        if (piece_val >= static_cast<int8_t> (Piece::LAKE) ||
                        piece_val == -static_cast<int8_t> (Piece::LAKE)) {
                            surrounded_count++;
                        }
                        break;
                    } else if (last_piece == Piece::SCOUT &&
                    board_view[check_pos[0] * width_ + check_pos[1]] <=
                    -static_cast<int8_t> (Piece::SPY)) {
                        surrounded_count++;
                        break;
                    }

                    if (last_piece != Piece::SCOUT) {
                        if (restricted_mask[check_pos[0] * width_ + check_pos[1]]) {
                            surrounded_count++;
                        }
                        break;
                    }

                    check_pos = { static_cast<int8_t> (check_pos[0] + dir.first),
                        static_cast<int8_t> (check_pos[1] + dir.second) };
                }

                if (check_pos[0] < 0 || check_pos[0] >= height_ ||
                check_pos[1] < 0 || check_pos[1] >= width_) {
                    surrounded_count++;
                }
            }

            if (surrounded_count == 4) {
                surrounded[last_pos[0] * width_ + last_pos[1]] = true;
            }
        }
    }

    // Create final action mask
    for (size_t i = 0; i < height_; ++i) {
        for (size_t j = 0; j < width_; ++j) {
            if (board_view[i * width_ + j] >= static_cast<int8_t> (Piece::SPY) &&
            !surrounded[i * width_ + j]) {
                action_mask[i * width_ + j] = true;
            }
        }
    }
}

void StrategoEnv::valid_destinations (std::vector<bool>& action_mask) const {
    action_mask.clear ();
    action_mask.resize (height_ * width_, false);

    if (game_phase_ != GamePhase::MOVE) {
        return;
    }

    const Pos& selected       = player_state (current_player_).last_selected ();
    int8_t selected_piece_val = board_[selected[0] * width_ + selected[1]];

    // Initialize masks
    std::vector<bool> two_square_mask (height_ * width_, true);
    std::vector<bool> chasing_mask (height_ * width_, true);

    // Apply two-square rule restrictions
    auto two_square_valid = two_square_detector_.validate_select (
    current_player_, static_cast<Piece> (selected_piece_val), selected);

    if (!two_square_valid.first) {
        const auto& [start_pos, end_pos] = two_square_valid.second;

        if (start_pos == end_pos) {
            two_square_mask[start_pos[0] * width_ + start_pos[1]] = false;
        } else {
            if (start_pos[0] == end_pos[0]) { // Horizontal line
                int min_col = std::min (start_pos[1], end_pos[1]);
                int max_col = std::max (start_pos[1], end_pos[1]);
                for (int j = min_col; j <= max_col; ++j) {
                    two_square_mask[start_pos[0] * width_ + j] = false;
                }
            } else { // Vertical line
                int min_row = std::min (start_pos[0], end_pos[0]);
                int max_row = std::max (start_pos[0], end_pos[0]);
                for (int i = min_row; i <= max_row; ++i) {
                    two_square_mask[i * width_ + start_pos[1]] = false;
                }
            }
        }
    }

    // Apply chasing rule restrictions
    Pos rotated_selected = selected;
    if (current_player_ == Player::BLUE) {
        rotated_selected = rotate_coord (selected);
    }

    auto chasing_valid = chasing_detector_.validate_select (current_player_,
    static_cast<Piece> (selected_piece_val), rotated_selected, board_, height_, width_);

    if (!chasing_valid.first) {
        for (const Pos& pos : chasing_valid.second) {
            Pos actual_pos = pos;
            if (current_player_ == Player::BLUE) {
                actual_pos = rotate_coord (pos);
            }
            chasing_mask[actual_pos[0] * width_ + actual_pos[1]] = false;
        }
    }

    // Generate possible destinations based on piece type
    const std::array<std::pair<int, int>, 4> directions = { {
    { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } // right, left, down, up
    } };

    if (selected_piece_val == static_cast<int8_t> (Piece::SCOUT)) {
        // Scout can move multiple squares in one direction
        for (const auto& dir : directions) {
            Pos current_pos       = selected;
            int encountered_enemy = 0;

            while (true) {
                // Update position
                current_pos[0] += static_cast<int8_t> (dir.first);
                current_pos[1] += static_cast<int8_t> (dir.second);

                // Check boundaries
                if (current_pos[0] < 0 || current_pos[0] >= height_ ||
                current_pos[1] < 0 || current_pos[1] >= width_) {
                    break;
                }

                // Check piece
                int8_t target_val = board_[current_pos[0] * width_ + current_pos[1]];

                if (target_val != static_cast<int8_t> (Piece::EMPTY)) {
                    if (target_val > -static_cast<int8_t> (Piece::FLAG)) {
                        break;
                    }
                    encountered_enemy++;
                }

                // Stop if encountered more than 1 enemy
                if (encountered_enemy > 1) {
                    break;
                }

                // Mark as valid destination
                action_mask[current_pos[0] * width_ + current_pos[1]] = true;
            }
        }
    } else {
        // Regular pieces can only move one square
        for (const auto& dir : directions) {
            Pos dest = { static_cast<int8_t> (selected[0] + dir.first),
                static_cast<int8_t> (selected[1] + dir.second) };

            // Check boundaries
            if (dest[0] < 0 || dest[0] >= height_ || dest[1] < 0 || dest[1] >= width_) {
                continue;
            }

            int8_t target_val = board_[dest[0] * width_ + dest[1]];

            // Valid if empty or enemy piece (not lake)
            if (target_val <= static_cast<int8_t> (Piece::EMPTY) &&
            target_val != -static_cast<int8_t> (Piece::LAKE)) {
                action_mask[dest[0] * width_ + dest[1]] = true;
            }
        }
    }

    // Apply restrictions from special rules
    for (size_t i = 0; i < height_; ++i) {
        for (size_t j = 0; j < width_; ++j) {
            size_t idx = i * width_ + j;
            if (!two_square_mask[idx] || !chasing_mask[idx]) {
                action_mask[idx] = false;
            }
        }
    }
}

Pos StrategoEnv::get_random_action () const {
    std::random_device rd;
    std::mt19937 gen (rd ());

    std::vector<bool> action_mask;

    switch (game_phase_) {
    case GamePhase::DEPLOY: {
        valid_spots_to_place (action_mask);
        break;
    }
    case GamePhase::SELECT: {
        valid_pieces_to_select (action_mask);
        break;
    }
    case GamePhase::MOVE: {
        valid_destinations (action_mask);
        break;
    }
    default: return { -1, -1 };
    }

    std::vector<size_t> indices;
    for (size_t i = 0; i < action_mask.size (); ++i) {
        if (action_mask[i])
            indices.push_back (i);
    }
    if (indices.empty ())
        return { -1, -1 };
    std::uniform_int_distribution<> dist (0, indices.size () - 1);
    size_t idx = indices[dist (gen)];
    return { static_cast<int8_t> (idx / width_), static_cast<int8_t> (idx % width_) };
}

inline const PlayerStateHandler& StrategoEnv::player_state (Player player, bool opponent) const {
    return current_player_ == Player::RED ^ opponent ? p1_ : p2_;
}

inline Pos StrategoEnv::rotate_coord (const Pos& pos) const {
    return { static_cast<int8_t> (height_ - pos[0] - 1),
        static_cast<int8_t> (width_ - pos[1] - 1) };
}

const Pos& StrategoEnv::last_selected (Player player) const {
    return player_state (player).last_selected ();
}

inline void StrategoEnv::switch_current_player () {
    current_player_ = (current_player_ == Player::RED) ? Player::BLUE : Player::RED;
}
