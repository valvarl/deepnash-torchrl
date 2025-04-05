#include "stratego_env.h"
#include <stdexcept>
#include <random>
#include <algorithm>
#include <numeric>

void roll(std::vector<double>& vec, int shift) {
    if (vec.empty()) return;
    
    shift %= static_cast<int>(vec.size());
    if (shift < 0) {
        shift += vec.size(); // обработка отрицательного сдвига
    }
    
    std::rotate(vec.begin(), vec.begin() + (vec.size() - shift), vec.end());
}

void rotate_plane(std::vector<double>& vec) {
    if (vec.empty()) return;
    std::reverse(vec.begin(), vec.end());
    for (auto& elem : vec) {
        elem *= -1;
    }
}

StrategoEnv::StrategoEnv(std::shared_ptr<StrategoConfig> config, uint32_t seed)
    : config_(std::move(config)),
      game_phase_(GamePhase::TERMINAL),
      height_(config_->height()),
      width_(config_->width()),
      board_(height_, width_, static_cast<int8_t>(Piece::EMPTY)),
      lakes_(height_, width_, false),
      p1_(Player::RED),
      p2_(Player::BLUE),
      action_space_({0, 0})
{
    if (!config_) {
        config_ = std::make_shared<StrategoConfig>(StrategoConfig::from_game_mode(GameMode::ORIGINAL));
    }

    action_space_ = MaskedMultiDiscrete({static_cast<int>(height_), static_cast<int>(width_)}, seed);
}

void StrategoEnv::reset(uint32_t seed) {
    if (height_ != config_->height() || width_ != config_->width()) {
        height_ = config_->height();
        width_ = config_->width();
        action_space_ = MaskedMultiDiscrete({static_cast<int>(height_), static_cast<int>(width_)}, seed);
    }

    game_phase_ = GamePhase::DEPLOY;
    current_player_ = Player::RED;
    board_ = Matrix<int8_t>(height_, width_, static_cast<int8_t>(Piece::EMPTY)),
    lakes_ = config_->lakes_mask();

    total_moves_limit_ = config_->total_moves_limit();
    moves_since_attack_limit_ = config_->moves_since_attack_limit();
    observed_history_entries_ = config_->observed_history_entries();

    total_moves_ = 0;
    moves_since_attack_ = 0;

    allowed_pieces_ = config_->allowed_pieces();

    board_ = Matrix<int8_t>(height_, width_, static_cast<int8_t>(Piece::EMPTY));
    lakes_ = config_->lakes_mask();
    for (size_t i = 0; i < height_; ++i) {
        for (size_t j = 0; j < width_; ++j) {
            if (lakes_(i, j)) {
                board_(i, j) = static_cast<int8_t>(Piece::LAKE);
            }
        }
    }

    std::vector<int> p1_pieces_vec(static_cast<int>(Piece::MARSHAL) + 1, 0);
    for (const auto& [piece, count] : config_->p1_pieces()) {
        p1_pieces_vec[static_cast<int>(piece)] = count;
    }
    
    std::vector<int> p2_pieces_vec(static_cast<int>(Piece::MARSHAL) + 1, 0);
    for (const auto& [piece, count] : config_->p2_pieces()) {
        p2_pieces_vec[static_cast<int>(piece)] = count;
    }
    
    p1_.generate_state(p1_pieces_vec, config_->p1_deploy_mask(), 
                      observed_history_entries_, height_, width_);
    p2_.generate_state(p2_pieces_vec, config_->p2_deploy_mask(),
                      observed_history_entries_, height_, width_);
    
    // Сброс счётчиков
    two_square_detector_.reset();
    chasing_detector_.reset();
}

void StrategoEnv::generate_observation(std::vector<double> &obs) const {
    obs.reserve(allowed_pieces_.size() * 3 + observed_history_entries_ + 6);

    // 1. Lakes
    for (size_t i = 0; i < height_; ++i) {
        for (size_t j = 0; j < width_; ++j) {
            obs.push_back(lakes_(i, j) ? 1.0 : 0.0);
        }
    }

    // 2. Private observation
    for (int piece_val : allowed_pieces_) {
        for (size_t i = 0; i < height_; ++i) {
            for (size_t j = 0; j < width_; ++j) {
                int8_t val = board_(i, j);
                bool active_piece = val > static_cast<int8_t>(Piece::LAKE);
                obs.push_back((active_piece && val == piece_val) ? 1.0 : 0.0);
            }
        }
    }

    // 3. Public observation and move history
    if (game_phase_ == GamePhase::DEPLOY) {
        obs.resize(obs.size() + (allowed_pieces_.size() * 2 + observed_history_entries_) * height_ * width_, 0.0);
    } else {
        std::vector<double> public_obs, opp_public_obs, move_matrix;
        const PlayerStateHandler& cur = current_player_ == Player::RED ? p1_ : p2_;
        const PlayerStateHandler& opp = current_player_ == Player::RED ? p2_ : p1_;
        
        get_public_obs(cur.public_obs_info(), cur.unrevealed(), cur.pieces(), cur.movable_pieces(), public_obs);
        get_public_obs(opp.public_obs_info(), opp.unrevealed(), opp.pieces(), opp.movable_pieces(), opp_public_obs);
        (current_player_ == Player::RED ? p1_ : p2_).observed_moves();
        
        obs.emplace_back(public_obs);
        obs.emplace_back(opp_public_obs);
        obs.emplace_back(move_matrix);
    }

    // 4. Scalar info
    double total_moves_ratio = static_cast<double>(total_moves_) / total_moves_limit_;
    double since_attack_ratio = static_cast<double>(moves_since_attack_) / moves_since_attack_limit_;

    for (size_t i = 0; i < height_; ++i) {
        for (size_t j = 0; j < width_; ++j) {
            obs.push_back(total_moves_ratio);
            obs.push_back(since_attack_ratio);
            obs.push_back(game_phase_ == GamePhase::DEPLOY ? 1.0 : 0.0);
            obs.push_back(game_phase_ == GamePhase::MOVE ? 1.0 : 0.0);
        }
    }

    // 5. last_selected
    const Pos& last_selected = current_player_ == Player::RED ? p1_.last_selected() : p2_.last_selected();
    for (size_t i = 0; i < height_; ++i) {
        for (size_t j = 0; j < width_; ++j) {
            obs.push_back((i == last_selected[0] && j == last_selected[1]) ? 1.0 : 0.0);
        }
    }
}

void StrategoEnv::generate_env_state(
    std::vector<double>& obs,
    std::vector<bool>& action_mask
) {
    generate_observation(obs);
    if (game_phase_ == GamePhase::DEPLOY) {
        valid_spots_to_place(action_mask);
    } else if (game_phase_ == GamePhase::SELECT) {
        valid_pieces_to_select(action_mask);
    } else {
        valid_destinations(action_mask);
    }
    action_space_.set_mask(action_mask);
}

void StrategoEnv::get_public_obs(
    const std::array<Matrix<bool>, 3>& public_obs_info,
    const std::vector<int>& unrevealed,
    const std::vector<int>& pieces,
    const std::vector<int>& movable_pieces,
    std::vector<double>& public_obs
) const
{
    size_t num_pieces = pieces.size();
    public_obs.reserve(num_pieces * height_ * width_);

    // Суммы для нормализации
    int sum_all = 0;
    for (int p : pieces) sum_all += unrevealed[p];

    int sum_movable = 0;
    for (int p : movable_pieces) sum_movable += unrevealed[p];

    // Расчёт вероятностей
    std::vector<double> probs_unmoved(num_pieces, 0.0);
    std::vector<double> probs_moved(num_pieces, 0.0);

    for (size_t i = 0; i < num_pieces; ++i) {
        int piece = pieces[i];
        if (sum_all > 0)
            probs_unmoved[i] = static_cast<double>(unrevealed[piece]) / sum_all;

        if (sum_movable > 0 && std::find(movable_pieces.begin(), movable_pieces.end(), piece) != movable_pieces.end()) {
            probs_moved[i] = static_cast<double>(unrevealed[piece]) / sum_movable;
        }
    }

    for (size_t i = 0; i < num_pieces; ++i) {
        for (size_t r = 0; r < height_; ++r) {
            for (size_t c = 0; c < width_; ++c) {
                double val_unmoved = public_obs_info[0](r, c) ? probs_unmoved[i] : 0.0;
                double val_moved = public_obs_info[1](r, c) ? probs_moved[i] : 0.0;
                double val_revealed = public_obs_info[2](r, c) && static_cast<int>(pieces[i]) == static_cast<int>(public_obs_info[2](r, c)) ? 1.0 : 0.0;
                public_obs.push_back(val_unmoved + val_moved + val_revealed);
            }
        }
    }
}

void StrategoEnv::encode_move(const Pos& src, const Pos& dest, std::vector<double>& encoding) const {
    encoding.resize(height_ * width_, 0.0);
    int8_t src_piece = board_(src[0], src[1]);
    int8_t dest_piece = board_(dest[0], dest[1]);

    if (dest_piece == static_cast<int8_t>(Piece::EMPTY)) {
        encoding[dest[0] * width_ + dest[1]] = 1.0;
        encoding[src[0] * width_ + src[1]] = -1.0;
    } else {
        double weight = 2.0 + (static_cast<double>(src_piece) - 3.0) / 12.0;
        encoding[dest[0] * width_ + dest[1]] = 1.0;
        encoding[src[0] * width_ + src[1]] = -weight;
    }
}

std::tuple<std::vector<double>, std::vector<bool>, int, bool, bool> StrategoEnv::step(const Pos& action) {
    std::pair<bool, std::string> action_valid = validate_coord(action);
    if (!action_valid.first) {
        throw std::invalid_argument(action_valid.second);
    }
    
    int reward = 0;
    bool terminated = false;
    bool truncated = false;
    
    std::vector<double> obs;
    std::vector<bool> action_mask;

    switch (game_phase_) {
        case GamePhase::DEPLOY: {
            valid_spots_to_place(action_mask);
            if (!action_mask[action[0] * width_ + action[1]]) {
                // action = action_space_.sample();
                throw std::invalid_argument("Invalid deployment location");
            }
            
            auto& curr_player = (current_player_ == Player::RED) ? p1_ : p2_;
            auto& opp_player = (current_player_ == Player::RED) ? p2_ : p1_;
            size_t deploy_idx = 0, deploy_piece = 0;
            for (auto piece : allowed_pieces_) {
                int i = 0;
                for (; i < curr_player.unrevealed_[piece]; ++i) {
                    if (deploy_idx == curr_player.deploy_idx_) {
                        deploy_piece = piece;
                        break;
                    }
                }
                if (deploy_piece != 0) {
                    break;
                } 
                deploy_idx += i;
            }
            
            board_(action[0], action[1]) = deploy_piece;
            ++curr_player.deploy_idx_;

            bool curr_finish_deploy = curr_player.deploy_idx_ == std::accumulate(curr_player.unrevealed_.begin(), curr_player.unrevealed_.end(), 0);
            bool opp_finish_deploy = opp_player.deploy_idx_ == std::accumulate(opp_player.unrevealed_.begin(), opp_player.unrevealed_.end(), 0);
            if (opp_finish_deploy && !curr_finish_deploy) {
                break;
            } else if (curr_finish_deploy && opp_finish_deploy) {
                std::vector<bool> action_mask;
                valid_pieces_to_select(action_mask, true);
                bool opp_no_moves = std::any_of(action_mask.begin(), action_mask.end(), [](bool x) { return x; });
                if (opp_no_moves) {
                    action_mask.clear();
                    valid_pieces_to_select(action_mask, false);
                    bool draw_game = std::any_of(action_mask.begin(), action_mask.end(), [](bool x) { return x; });
                    game_phase_ = GamePhase::TERMINAL;
                    reward = draw_game ? 0 : 1;
                    terminated = true;
                    break;
                }
                game_phase_ = GamePhase::SELECT;
                for (size_t r = 0; r < height_; ++r) {
                    for (size_t c = 0; c < width_; ++c) {
                        if (board_(r, c) > static_cast<int8_t>(Piece::LAKE) && curr_player.deploy_mask_(r, c)) {
                            curr_player.public_obs_info_[0](r, c) = true;
                        }
                        size_t opp_r = height_ - r - 1, opp_c = width_ - c - 1;
                        if (board_(opp_r, opp_c) < -static_cast<int8_t>(Piece::LAKE) && opp_player.deploy_mask_(opp_r, opp_c)) {
                            opp_player.public_obs_info_[0](r, c) = true;
                        }
                    }
                }
            }
            rotate_board(board_);
            current_player_ = (current_player_ == Player::RED) ? Player::BLUE : Player::RED;
            break;
        }
        case GamePhase::SELECT: {
            valid_pieces_to_select(action_mask);
            if (!action_mask[action[0] * width_ + action[1]]) {
                // action = action_space_.sample();
                throw std::invalid_argument("Invalid piece selection");
            }
            
            auto& player = (current_player_ == Player::RED) ? p1_ : p2_;
            player.set_last_selected(action);
            player.set_last_selected_piece(static_cast<Piece>(board_(action[0], action[1])));
            game_phase_ = GamePhase::MOVE;
            break;
        }
        case GamePhase::MOVE: {
            auto src = (current_player_ == Player::RED) ? p1_.last_selected() : p2_.last_selected();
            auto dest = action;
            
            if (!check_action_valid(src, dest)) {
                // action = action_space_.sample();
                // dest = action;
                throw std::invalid_argument("Invalid move");
            }

            auto& curr_player = (current_player_ == Player::RED) ? p1_ : p2_;
            auto& opp_player = (current_player_ == Player::RED) ? p2_ : p1_;

            curr_player.last_selected_ = action;

            int8_t src_piece_val = board_(src[0], src[1]);
            int8_t dest_piece_val = board_(dest[0], dest[1]);
            
            // Check if draw conditions are met
            if (total_moves_ >= total_moves_limit_ || 
                moves_since_attack_ >= moves_since_attack_limit_) {
                terminated = true;
                rotate_board(board_);
                current_player_ = (current_player_ == Player::RED) ? Player::BLUE : Player::RED;
                game_phase_ = GamePhase::TERMINAL;
                break;
            }

            // Update Draw conditions
            total_moves_++;
            moves_since_attack_ = (dest_piece_val == static_cast<int8_t>(Piece::EMPTY)) ? moves_since_attack_ + 1 : 0;
            
            // Update Move Histories
            roll(curr_player.observed_moves_, height_ * width_);
            roll(opp_player.observed_moves_, height_ * width_);
            std::vector<double> move;
            encode_move(src, dest, move);
            std::copy(move.begin(), move.end(), curr_player.observed_moves_.begin());
            rotate_plane(move);
            std::copy(move.begin(), move.end(), opp_player.observed_moves_.begin());

            two_square_detector_.update(current_player_, static_cast<Piece>(src_piece_val), src, dest);
            Pos src_rot = {static_cast<int8_t>(height_ - src[0] - 1), static_cast<int8_t>(width_ - src[1] - 1)};
            Pos dest_rot = {static_cast<int8_t>(height_ - dest[0] - 1), static_cast<int8_t>(width_ - dest[1] - 1)};
            
            auto _src = src, _dest = dest;
            if (current_player_ == Player::BLUE) {
                _src = src_rot;
                _dest = dest_rot;
            }
            chasing_detector_.update(current_player_, static_cast<Piece>(src_piece_val), _src, _dest, board_);

            if (src_piece_val = -dest_piece_val) { // Equal Strength
                // remove both pieces
                board_(src[0], src[1]) = 0;
                board_(dest[0], dest[1]) = 0;
                for (int k = 0; k < 3; ++k) {
                    curr_player.public_obs_info_[k](src[0], src[1]) = 0;
                    curr_player.public_obs_info_[k](dest[0], dest[1]) = 0;
                    opp_player.public_obs_info_[k](src_rot[0], src_rot[1]) = 0;
                    opp_player.public_obs_info_[k](dest_rot[0], dest_rot[1]) = 0;
                }
                curr_player.unrevealed_[src_piece_val]--;
                opp_player.unrevealed_[dest_piece_val]--;
            } else if ((src_piece_val == static_cast<int8_t>(Piece::SPY) && dest_piece_val == -static_cast<int8_t>(Piece::MARSHAL)) ||  // Spy vs Marshal
                       (src_piece_val > -dest_piece_val && (src_piece_val == static_cast<int8_t>(Piece::MINER) && dest_piece_val == -static_cast<int8_t>(Piece::BOMB) || 
                        dest_piece_val != -static_cast<int8_t>(Piece::BOMB))) ||  // attacker is stronger (+Bomb case)
                       (dest_piece_val == -static_cast<int8_t>(Piece::FLAG))) {  // enemy Flag found
                // remove enemy piece
                board_(src[0], src[1]) = 0;
                board_(dest[0], dest[1]) = src_piece_val;
                for (int k = 0; k < 3; ++k) {
                    curr_player.public_obs_info_[k](src[0], src[1]) = 0;
                }
                if (dest_piece_val != static_cast<int8_t>(Piece::EMPTY)) {
                    curr_player.public_obs_info_[2](dest[0], dest[1]) = src_piece_val;
                    for (int k = 0; k < 3; ++k) {
                        opp_player.public_obs_info_[k](dest_rot[0], dest_rot[1]) = 0;
                    }
                    curr_player.unrevealed_[src_piece_val]--;
                    opp_player.unrevealed_[dest_piece_val]--;
                } else {
                    bool scout_move = (src[0] == dest[0] && abs(src[1] - dest[1]) > 1) || (src[1] == dest[1] && abs(src[0] - dest[0]) > 1);
                    if (scout_move) {
                        curr_player.public_obs_info_[2](dest[0], dest[1]) = src_piece_val;
                        curr_player.unrevealed_[src_piece_val]--;
                    } else {
                        curr_player.public_obs_info_[1](dest[0], dest[1]) = 1;
                    }
                }
                if (dest_piece_val == -static_cast<int8_t>(Piece::FLAG)) {
                    reward = 1;
                    terminated = true;
                }
            } else if (src_piece_val < -dest_piece_val || dest_piece_val == -static_cast<int8_t>(Piece::BOMB)) {
                // remove attacker
                board_(src[0], src[1]) = 0;
                for (int k = 0; k < 3; ++k) {
                    curr_player.public_obs_info_[k](src[0], src[1]) = 0;
                }
                for (int k = 0; k < 2; ++k) {
                    opp_player.public_obs_info_[k](dest_rot[0], dest_rot[1]) = 0;
                }
                opp_player.public_obs_info_[2](dest_rot[0], dest_rot[1]) = dest_piece_val;
                curr_player.unrevealed_[src_piece_val]--;
                opp_player.unrevealed_[dest_piece_val]--;
            }
            
            rotate_board(board_);
            current_player_ = (current_player_ == Player::RED) ? Player::BLUE : Player::RED;

            // Check if any pieces can be moved. If one player has no movable pieces, the other player wins.
            // If both players have no movable pieces, the game is a draw.
            if (!terminated) {
                std::vector<bool> action_mask;
                valid_pieces_to_select(action_mask, false);
                bool curr_no_moves = std::any_of(action_mask.begin(), action_mask.end(), [](bool x) { return x; });
                if (curr_no_moves) {
                    action_mask.clear();
                    valid_pieces_to_select(action_mask, true);
                    bool draw_game = std::any_of(action_mask.begin(), action_mask.end(), [](bool x) { return x; });
                    game_phase_ = GamePhase::TERMINAL;
                    reward = draw_game ? 0 : 1;
                    terminated = true;
                    break;
                }
            }
            game_phase_ = terminated ? GamePhase::TERMINAL : GamePhase::SELECT;
            break;
        }
        
        case GamePhase::TERMINAL:
            throw std::runtime_error("Game has already terminated");
    }
    
    generate_env_state(obs, action_mask);
    return {std::move(obs), std::move(action_mask), reward, terminated, truncated};
}

inline std::pair<bool, std::string> StrategoEnv::validate_coord(const Pos& coord) const {
    if (coord[0] < 0 or coord[0] >= height_) {
        return {false, "Source row is out of bounds"};
    } else if (coord[1] < 0 or coord[1] >= width_) {
        return {false, "Source column is out of bounds"};
    }
    return {true, ""};
}

bool StrategoEnv::check_action_valid(const Pos& src, const Pos& dest) const {
    // Проверка координат
    if (src[0] < 0 || src[0] >= static_cast<int>(config_->height()) ||
        src[1] < 0 || src[1] >= static_cast<int>(config_->width()) ||
        dest[0] < 0 || dest[0] >= static_cast<int>(config_->height()) ||
        dest[1] < 0 || dest[1] >= static_cast<int>(config_->width())) {
        return false;
    }
    
    Piece selected_piece = static_cast<Piece>(board_(src[0], src[1]));
    Piece destination = static_cast<Piece>(board_(dest[0], dest[1]));
    
    // Основные проверки допустимости хода
    if (selected_piece < Piece::SPY) return false;
    if (destination == Piece::LAKE) return false;
    if (destination > Piece::LAKE) return false;
    
    // Проверка движения для разных типов фигур
    // ...
    
    return true;
}

void StrategoEnv::valid_spots_to_place(std::vector<bool>& action_mask) const {
    // const auto& deploy_mask = (current_player_ == Player::RED) ? 
    //     p1_.deploy_mask() : p2_.deploy_mask();
    
    // for (size_t y = 0; y < height_; ++y) {
    //     for (size_t x = 0; x < width_; ++x) {
    //         if (board_(y, x) == static_cast<int8_t>(Piece::EMPTY) && deploy_mask(y, x)) {
    //             action_mask(y, x) = true;
    //         }
    //     }
    // }
}

void StrategoEnv::valid_pieces_to_select(std::vector<bool>& action_mask, bool is_other_player = false) const {

}

void StrategoEnv::valid_destinations(std::vector<bool>& action_mask) const {

}

Pos StrategoEnv::get_random_action() const {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<bool> action_mask; 

    switch (game_phase_) {
        case GamePhase::DEPLOY: {
            valid_spots_to_place(action_mask);
            break;
        }
        case GamePhase::SELECT: {
            valid_pieces_to_select(action_mask);
            break;
        }
        case GamePhase::MOVE: {
            valid_destinations(action_mask);
            break;
        }
        default:
            return {-1, -1};
    }

    std::vector<size_t> indices;
    for (size_t i = 0; i < action_mask.size(); ++i) {
        if (action_mask[i]) indices.push_back(i);
    }
    if (indices.empty()) return {-1, -1};
    std::uniform_int_distribution<> dist(0, indices.size() - 1);
    size_t idx = indices[dist(gen)];
    return {static_cast<int8_t>(idx / width_), static_cast<int8_t>(idx % width_)};
}

void StrategoEnv::rotate_board(Matrix<int8_t>& board) {
    std::reverse(board.data_.begin(), board.data_.end());
    for (auto& elem : board.data_) {
        elem *= -1;
    }
}
