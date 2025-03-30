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
    std::tuple<std::vector<double>, int, bool, bool> step(const Pos& action);

private:
    // Observation and state generation
    const std::vector<double>& generate_observation() const;
    std::pair<const std::vector<double>&, const Matrix<bool>&> generate_env_state();

    // Observation helpers
    const std::vector<Matrix<double>>& get_public_obs(
        const std::array<Matrix<bool>, 3>& public_obs_info,
        const std::vector<int>& unrevealed,
        const std::vector<int>& pieces,
        const std::vector<int>& movable_pieces) const;

    const Matrix<double>& StrategoEnv::encode_move(const std::array<Matrix<bool>, 2>& action) const;

    // Movement and validation
    std::tuple<std::vector<double>, int, bool, bool> movement_step(const Pos& action);
    bool check_action_valid(const Pos& src, const Pos& dest) const;
    std::pair<bool, std::string> validate_coord(const Pos& coord) const;

    Matrix<bool> valid_spots_to_place() const;
    Matrix<bool> valid_pieces_to_select(bool is_other_player = false) const;
    Matrix<bool> valid_destinations() const;

    // Helper methods
    Pos get_random_action() const;

    // Getters
    const Matrix<int8_t>& board() const { return board_; }
    GamePhase game_phase() const { return game_phase_; }
    Player current_player() const { return current_player_; }
    const PlayerStateHandler& player_state(Player player) const;
};

#endif // STRATEGO_ENV_H