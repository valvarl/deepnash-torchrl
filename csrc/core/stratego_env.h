#ifndef STRATEGO_ENV_H
#define STRATEGO_ENV_H

#include "primitives.h"
#include "stratego_config.h"
#include "detectors.h"
#include "masked_multi_discrete.h"
#include "player_state_handler.h"

#include <vector>
#include <unordered_map>
#include <memory>
#include <array>
#include <optional>

class StrategoEnv {
private:
    // Configuration
    std::shared_ptr<StrategoConfig> config_;
    size_t height_;
    size_t width_;

    // Game state
    GamePhase game_phase_;
    Player current_player_;
    Matrix<int8_t> board_;
    Matrix<bool> lakes_;

    // Players
    PlayerStateHandler p1_;
    PlayerStateHandler p2_;

    // Detectors
    TwoSquareDetector two_square_detector_;
    ChasingDetector chasing_detector_;

    // Game counters
    int total_moves_limit_;
    int moves_since_attack_limit_;
    int observed_history_entries_;

    int total_moves_;
    int moves_since_attack_;

    std::vector<int> allowed_pieces_;

    // Action space
    MaskedMultiDiscrete action_space_;

public:
    StrategoEnv(std::shared_ptr<StrategoConfig> config = nullptr, uint32_t seed = 0);
    
    void reset(uint32_t seed = 0);
    std::tuple<std::vector<double>, std::vector<bool>, int, bool, bool> step(const Pos& action);

    // Getters
    const Matrix<int8_t>& board() const { return board_; }
    GamePhase game_phase() const { return game_phase_; }
    Player current_player() const { return current_player_; }
    const PlayerStateHandler& player_state(Player player) const;

private:
    // Observation and state generation
    void generate_observation(std::vector<double>& obs) const;
    void generate_env_state(std::vector<double>& obs, std::vector<bool>& action_mask);

    // Observation helpers
    void get_public_obs(
        const std::array<Matrix<bool>, 3>& public_obs_info,
        const std::vector<int>& unrevealed,
        const std::vector<int>& pieces,
        const std::vector<int>& movable_pieces,
        std::vector<double>& public_obs) const;

    void encode_move(const Pos& src, const Pos& dest, std::vector<double>& encoding) const;

    // Movement and validation
    std::tuple<std::vector<double>, int, bool, bool> movement_step(const Pos& action);
    bool check_action_valid(const Pos& src, const Pos& dest) const;
    std::pair<bool, std::string> validate_coord(const Pos& coord) const;

    void valid_spots_to_place(std::vector<bool>& action_mask) const;
    void valid_pieces_to_select(std::vector<bool>& action_mask, bool is_other_player = false) const;
    void valid_destinations(std::vector<bool>& action_mask) const;

    // Helper methods
    Pos get_random_action() const;
    void rotate_board(Matrix<int8_t>& board);
};

#endif // STRATEGO_ENV_H