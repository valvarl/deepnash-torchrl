#include "detectors.h"
#include "primitives.h"
#include <algorithm>

// ChasingDetector implementation
bool ChasingDetector::is_adjacent(const Pos& pos1, const Pos& pos2) const {
    return (abs(pos1[0] - pos2[0]) == 1 && pos1[1] == pos2[1]) ||
           (abs(pos1[1] - pos2[1]) == 1 && pos1[0] == pos2[0]);
}

bool ChasingDetector::is_on_same_line(const Pos& pos1, const Pos& pos2,
                                     const Matrix<int8_t>& board) const {
    if (pos1[0] == pos2[0]) {
        int min_col = std::min(pos1[1], pos2[1]);
        int max_col = std::max(pos1[1], pos2[1]);
        for (int col = min_col + 1; col < max_col; ++col) {
            if (board(pos1[0], col) != static_cast<int8_t>(Piece::EMPTY)) {
                return false;
            }
        }
        return true;
    }
    if (pos1[1] == pos2[1]) {
        int min_row = std::min(pos1[0], pos2[0]);
        int max_row = std::max(pos1[0], pos2[0]);
        for (int row = min_row + 1; row < max_row; ++row) {
            if (board(row, pos1[1]) != static_cast<int8_t>(Piece::EMPTY)) {
                return false;
            }
        }
        return true;
    }
    return false;
}

bool ChasingDetector::check_chasing_condition(Piece verified_piece,
                                            const Pos& verified_pos,
                                            const Pos& opponent_pos,
                                            const Matrix<int8_t>& board) const {
    if (verified_piece != Piece::SCOUT) {
        return is_adjacent(verified_pos, opponent_pos);
    } else {
        return is_on_same_line(verified_pos, opponent_pos, board);
    }
}

std::pair<bool, std::vector<Pos>> ChasingDetector::validate_select(
    Player player, Piece piece, const Pos& pos, const Matrix<int8_t>& board) const {
    
    if (chase_moves_.empty() || chase_moves_.back().attacker) {
        return {true, {}};
    }

    std::vector<Pos> forbidden;
    for (size_t i = 0; i < chase_moves_.size() / 2; ++i) {
        const auto& chasing_move = chase_moves_[i * 2];
        const auto& chased_move = chase_moves_[i * 2 + 1];

        if (chasing_move.player == player && 
            chasing_move.piece == piece &&
            chased_move.from_pos == chase_moves_.back().to_pos &&
            check_chasing_condition(piece, pos, chasing_move.to_pos, board)) {
            
            if ((chase_moves_.size() - 1) / 2 == i + 1 && 
                chase_moves_[chase_moves_.size() - 2].from_pos == chasing_move.to_pos) {
                continue;
            } else {
                forbidden.push_back(chasing_move.to_pos);
            }
        }
    }

    return forbidden.empty() ? std::make_pair(true, std::vector<Pos>{}) 
                            : std::make_pair(false, forbidden);
}

bool ChasingDetector::validate_move(Player player, Piece piece, 
                                  const Pos& from_pos, const Pos& to_pos,
                                  const Matrix<int8_t>& board) const {
    auto [valid, forbidden] = validate_select(player, piece, from_pos, board);
    if (valid) return true;
    return std::find(forbidden.begin(), forbidden.end(), to_pos) == forbidden.end();
}

void ChasingDetector::update(Player player, Piece piece,
                            const Pos& from_pos, const Pos& to_pos,
                            const Matrix<int8_t>& board) {
    if (chase_moves_.empty()) {
        chase_moves_.emplace_back(player, piece, from_pos, to_pos);
        return;
    }

    if (chase_moves_.size() > 1 && 
        chase_moves_[chase_moves_.size() - 2].to_pos != from_pos) {
        // Selection of a figure not involved in chasing
        chase_moves_ = {chase_moves_.back()};
        chase_moves_.back().attacker = true;
    } else if (chase_moves_.back().attacker && 
               check_chasing_condition(piece, to_pos, chase_moves_.back().to_pos, board)) {
        // Initiative handover
        chase_moves_ = {ChaseEntry(player, piece, from_pos, to_pos)};
        return;
    }

    if (check_chasing_condition(
        chase_moves_.back().attacker ? chase_moves_.back().piece : piece,
        chase_moves_.back().attacker ? to_pos : from_pos,
        chase_moves_.back().to_pos,
        board)) {
        
        // Chase continues
        if (!chase_moves_.back().attacker && 
            !validate_move(player, piece, from_pos, to_pos, board)) {
            throw std::runtime_error("Invalid chasing move");
        }
        chase_moves_.emplace_back(player, piece, from_pos, to_pos, !chase_moves_.back().attacker);
    } else {
        chase_moves_ = {ChaseEntry(player, piece, from_pos, to_pos)};
    }
}

// TwoSquareDetector implementation
const std::vector<std::pair<Pos, Pos>>& TwoSquareDetector::get_player_moves(Player player) const {
    return (player == Player::RED) ? p1_moves_ : p2_moves_;
}

std::pair<bool, std::pair<Pos, Pos>> TwoSquareDetector::validate_select(
    Player player, Piece piece, const Pos& pos) const {
    
    const auto& p = get_player_moves(player);
    if (p.size() < 3) {
        return {true, {}};
    }
    if (pos != p.back().second) {
        return {true, {}};
    }
    if (piece != Piece::SCOUT) {
        return {false, {p.back().first, p.back().first}};
    }

    auto [start_pos, end_pos] = p[0];
    if (start_pos[0] == end_pos[0]) {
        if (start_pos[1] < end_pos[1]) {
            start_pos[1] = std::min(start_pos[1], p[1].second[1]);
            end_pos[1] = std::max(end_pos[1], p[2].second[1]);
            return {false, {start_pos, pos}};
        } else {
            start_pos[1] = std::max(start_pos[1], p[1].second[1]);
            end_pos[1] = std::min(end_pos[1], p[2].second[1]);
            return {false, {pos, start_pos}};
        }
    } else {
        if (start_pos[0] < end_pos[0]) {
            start_pos[0] = std::min(start_pos[0], p[1].second[0]);
            end_pos[0] = std::max(end_pos[0], p[2].second[0]);
            return {false, {start_pos, pos}};
        } else {
            start_pos[0] = std::max(start_pos[0], p[1].second[0]);
            end_pos[0] = std::min(end_pos[0], p[2].second[0]);
            return {false, {pos, start_pos}};
        }
    }
}

bool TwoSquareDetector::validate_move(Player player, Piece piece,
                                    const Pos& from_pos, const Pos& to_pos) const {
    auto [valid, positions] = validate_select(player, piece, from_pos);
    if (valid) return true;

    auto [start_pos, end_pos] = positions;
    int idx = (start_pos[0] == end_pos[0]) ? 1 : 0;

    if (to_pos[1 - idx] == start_pos[1 - idx]) {
        if ((start_pos[idx] <= to_pos[idx] && to_pos[idx] <= end_pos[idx]) ||
            (end_pos[idx] <= to_pos[idx] && to_pos[idx] <= start_pos[idx])) {
            return false;
        }
    }
    return true;
}

void TwoSquareDetector::update(Player player, Piece piece,
                              const Pos& from_pos, const Pos& to_pos) {
    auto& p = (player == Player::RED) ? p1_moves_ : p2_moves_;

    if (p.empty()) {
        p.emplace_back(from_pos, to_pos);
        return;
    }

    if (from_pos != p.back().second) {
        p.clear();
        p.emplace_back(from_pos, to_pos);
        return;
    }

    auto [start_pos, end_pos] = p.back();
    int idx = (start_pos[0] == end_pos[0]) ? 1 : 0;

    if ((start_pos[idx] < end_pos[idx] && start_pos[idx] <= to_pos[idx] && to_pos[idx] < end_pos[idx]) ||
        (start_pos[idx] > end_pos[idx] && end_pos[idx] < to_pos[idx] && to_pos[idx] <= start_pos[idx])) {
        
        if (validate_move(player, piece, from_pos, to_pos)) {
            p.emplace_back(from_pos, to_pos);
        } else {
            throw std::runtime_error("Invalid two-square move");
        }
    } else {
        p.clear();
        p.emplace_back(from_pos, to_pos);
    }
}