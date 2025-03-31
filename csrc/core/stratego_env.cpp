#include "stratego_env.h"
#include <stdexcept>
#include <random>
#include <algorithm>
#include <numeric>

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

            auto selected_piece = board_(src[0], src[1]);
            auto destination = board_(dest[0], dest[1]);

            if (total_moves_ >= total_moves_limit_ || 
                moves_since_attack_ >= moves_since_attack_limit_) {
                terminated = true;
                rotate_board(board_);
                current_player_ = (current_player_ == Player::RED) ? Player::BLUE : Player::RED;
                game_phase_ = GamePhase::TERMINAL;
                break;
            }


            
            // Обновление истории движений
            // update_observation_history(src, dest);
            
            // Логика захвата и движения
            Piece attacker = static_cast<Piece>(board_(src[0], src[1]));
            Piece defender = static_cast<Piece>(board_(dest[0], dest[1]));
            // handle_capture(attacker, defender, src, dest);
            
            // Проверка условий победы
            if (defender == Piece::FLAG) {
                reward = 1;
                terminated = true;
            }
            
            // Обновление счётчиков
            total_moves_++;
            moves_since_attack_ = (defender == Piece::EMPTY) ? moves_since_attack_ + 1 : 0;
            
            // Проверка условий ничьи
            if (total_moves_ >= total_moves_limit_ || 
                moves_since_attack_ >= moves_since_attack_limit_) {
                terminated = true;
                truncated = true;
            }
            
            // Смена игрока и фазы
            current_player_ = (current_player_ == Player::RED) ? Player::BLUE : Player::RED;
            game_phase_ = terminated ? GamePhase::TERMINAL : GamePhase::SELECT;
            break;
        }
        
        case GamePhase::TERMINAL:
            throw std::runtime_error("Game has already terminated");
    }
    
    generate_env_state(obs, action_mask);
    return {std::move(obs), std::move(action_mask), reward, terminated, truncated};
}

std::tuple<std::vector<double>, int, bool, bool> StrategoEnv::movement_step(const Pos& dest) {
    // 1. Определить источник (source)
    Pos source = current_player_ == Player::RED ? p1_.last_selected() : p2_.last_selected();

    // 2. Проверить корректность хода
    if (!check_action_valid(source, dest)) {
        Pos sampled = action_space_.sample();
        return movement_step(sampled);  // Рекурсивный вызов с новым действием
    }

    // 3. Обновить last_selected
    if (current_player_ == Player::RED)
        p1_.set_last_selected(dest);
    else
        p2_.set_last_selected(dest);

    // 4. Получить фигуры
    int8_t selected_piece = board_(source[0], source[1]);
    int8_t destination = board_(dest[0], dest[1]);

    // 5. Создать action mask (2 канала)
    std::array<Matrix<bool>, 2> action_mask = {
        Matrix<bool>(height_, width_, false),
        Matrix<bool>(height_, width_, false)
    };
    action_mask[0](source[0], source[1]) = true;
    action_mask[1](dest[0], dest[1]) = true;

    // 6. Проверка на ничью
    if (total_moves_ >= total_moves_limit_ || moves_since_attack_ >= moves_since_attack_limit_) {
        rotate_board();
        game_phase_ = GamePhase::TERMINAL;
        return {generate_observation(), 0, true, false};
    }

    // 7. Обновление счётчиков
    total_moves_++;
    moves_since_attack_ = (destination == static_cast<int8_t>(Piece::EMPTY)) 
                          ? moves_since_attack_ + 1 : 0;

    // 8. Обновить историю ходов
    update_observation_history(source, dest);

    // 9. Обновить детекторы
    update_detectors(current_player_, selected_piece, source, dest);

    // 10. Боевая логика
    int reward = 0;
    bool terminated = false;

    if (selected_piece == -destination) {
        handle_capture_both(source, dest, selected_piece);
    } else if (
        (selected_piece == static_cast<int8_t>(Piece::SPY) && destination == -static_cast<int8_t>(Piece::MARSHAL)) ||
        (selected_piece > -destination && 
         (selected_piece == static_cast<int8_t>(Piece::MINER) && destination == -static_cast<int8_t>(Piece::BOMB)) ||
         destination != -static_cast<int8_t>(Piece::BOMB)) ||
        (destination == -static_cast<int8_t>(Piece::FLAG))
    ) {
        handle_capture_enemy(source, dest, selected_piece);
        if (destination == -static_cast<int8_t>(Piece::FLAG)) {
            reward = 1;
            terminated = true;
        }
    } else if (selected_piece < -destination || destination == -static_cast<int8_t>(Piece::BOMB)) {
        handle_capture_self(source, dest, selected_piece, destination);
    }

    // 11. Повернуть доску
    rotate_board();

    // 12. Проверка на конец игры из-за отсутствия ходов
    if (!terminated) {
        bool no_moves = !has_available_moves(current_player_);
        if (no_moves) {
            bool other_no_moves = !has_available_moves(get_opponent(current_player_));
            game_phase_ = GamePhase::TERMINAL;
            reward = other_no_moves ? 0 : 1;
            return {generate_observation(), reward, true, false};
        }
    }

    game_phase_ = terminated ? GamePhase::TERMINAL : GamePhase::SELECT;
    return {generate_observation(), reward, terminated, false};
}


Matrix<bool> StrategoEnv::valid_spots_to_place() const {
    Matrix<bool> mask(height_ * width_, false);
    const auto& deploy_mask = (current_player_ == Player::RED) ? 
        p1_.deploy_mask() : p2_.deploy_mask();
    
    for (size_t y = 0; y < height_; ++y) {
        for (size_t x = 0; x < width_; ++x) {
            if (board_(y, x) == static_cast<int8_t>(Piece::EMPTY) && deploy_mask(y, x)) {
                mask(y, x) = true;
            }
        }
    }
    return mask;
}

Matrix<bool> StrategoEnv::valid_pieces_to_select(bool is_other_player) const {
    // Аналогично Python-версии, но возвращает плоский вектор
    // ...
    return {}; // Заглушка
}

Matrix<bool> StrategoEnv::valid_destinations() const {
    // Аналогично Python-версии, но возвращает плоский вектор
    // ...
    return {}; // Заглушка
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

Pos StrategoEnv::get_random_action() const {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    switch (game_phase_) {
        case GamePhase::DEPLOY: {
            auto valid = valid_spots_to_place();
            std::vector<int> indices;
            for (size_t i = 0; i < valid.size(); ++i) {
                if (valid[i]) indices.push_back(i);
            }
            if (indices.empty()) return {-1, -1};
            std::uniform_int_distribution<> dist(0, indices.size() - 1);
            int idx = indices[dist(gen)];
            return {idx / config_->width(), idx % config_->width()};
        }
        
        case GamePhase::SELECT: {
            auto valid = valid_pieces_to_select();
            std::vector<int> indices;
            for (size_t i = 0; i < valid.size(); ++i) {
                if (valid[i]) indices.push_back(i);
            }
            if (indices.empty()) return {-1, -1};
            std::uniform_int_distribution<> dist(0, indices.size() - 1);
            int idx = indices[dist(gen)];
            return {idx / config_->width(), idx % config_->width()};
        }
        
        case GamePhase::MOVE: {
            auto valid = valid_destinations();
            std::vector<int> indices;
            for (size_t i = 0; i < valid.size(); ++i) {
                if (valid[i]) indices.push_back(i);
            }
            if (indices.empty()) return {-1, -1};
            std::uniform_int_distribution<> dist(0, indices.size() - 1);
            int idx = indices[dist(gen)];
            return {idx / config_->width(), idx % config_->width()};
        }
        
        default:
            return {-1, -1};
    }
}

void StrategoEnv::rotate_board(Matrix<int8_t>& board) {
    std::reverse(board.data_.begin(), board.data_.end());
}
