#ifndef PLAYER_STATE_HANDLER_H
#define PLAYER_STATE_HANDLER_H

#include "primitives.h"
#include <vector>
#include <array>
#include <memory>

class StrategoEnv;

class PlayerStateHandler {
friend class StrategoEnv;
private:
    Player player_;
    std::vector<int> pieces_;
    std::vector<int> movable_pieces_;
    size_t deploy_idx_;
    Matrix<bool> deploy_mask_;
    std::array<Matrix<bool>, 3> public_obs_info_;
    std::vector<int> unrevealed_;
    std::vector<double> observed_moves_;
    Pos last_selected_;
    Piece last_selected_piece_;

public:
    explicit PlayerStateHandler(Player player, 
                              size_t height = 0, 
                              size_t width = 0);
    
    void generate_state(
        const std::vector<int>& pieces_num,
        const Matrix<bool>& deploy_mask,
        size_t observed_history_entries,
        size_t height,
        size_t width
    );

    // Getters and setters
    Player player() const { return player_; }
    const std::vector<int>& pieces() const { return pieces_; }
    const std::vector<int>& movable_pieces() const { return movable_pieces_; }
    size_t deploy_idx() const { return deploy_idx_; }
    void set_deploy_idx(size_t idx) { deploy_idx_ = idx; }
    const Matrix<bool>& deploy_mask() const { return deploy_mask_; }
    const std::array<Matrix<bool>, 3>& public_obs_info() const { return public_obs_info_; }
    const std::vector<int>& unrevealed() const { return unrevealed_; }
    const std::vector<double>& observed_moves() const { return observed_moves_; }
    const Pos& last_selected() const { return last_selected_; }
    Piece last_selected_piece() const { return last_selected_piece_; }
    
    void set_last_selected(const Pos& pos) { last_selected_ = pos; }
    void set_last_selected_piece(Piece piece) { last_selected_piece_ = piece; }
};

#endif  // PLAYER_STATE_HANDLER_H