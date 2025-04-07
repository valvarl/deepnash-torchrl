#ifndef DETECTORS_H
#define DETECTORS_H

#include <memory>
#include <utility>
#include <vector>

#include "primitives.h"


class StrategoEnv;

struct ChaseEntry {
    Player player;
    Piece piece;
    Pos from_pos;
    Pos to_pos;
    bool attacker;

    ChaseEntry (Player p, Piece pc, Pos from, Pos to, bool att = true)
    : player (p), piece (pc), from_pos (from), to_pos (to), attacker (att) {
    }
};

class ChasingDetector {
    friend class StartegoEnv;

    private:
    std::vector<ChaseEntry> chase_moves_;

    bool is_adjacent (const Pos& pos1, const Pos& pos2) const;
    bool is_on_same_line (const Pos& pos1,
    const Pos& pos2,
    const std::vector<int8_t>& board,
    size_t height,
    size_t width) const;

    public:
    ChasingDetector () = default;

    bool check_chasing_condition (Piece verified_piece,
    const Pos& verified_pos,
    const Pos& opponent_pos,
    const std::vector<int8_t>& board,
    size_t height,
    size_t width) const;

    std::pair<bool, std::vector<Pos>> validate_select (Player player,
    Piece piece,
    const Pos& pos,
    const std::vector<int8_t>& board,
    size_t height,
    size_t width) const;

    bool validate_move (Player player,
    Piece piece,
    const Pos& from_pos,
    const Pos& to_pos,
    const std::vector<int8_t>& board,
    size_t height,
    size_t width) const;

    void update (Player player,
    Piece piece,
    const Pos& from_pos,
    const Pos& to_pos,
    const std::vector<int8_t>& board,
    size_t height,
    size_t width);

    void reset () {
        chase_moves_.clear ();
    }
};

class TwoSquareDetector {
    friend class StartegoEnv;

    private:
    std::vector<std::pair<Pos, Pos>> p1_moves_;
    std::vector<std::pair<Pos, Pos>> p2_moves_;

    const std::vector<std::pair<Pos, Pos>>& get_player_moves (Player player) const;

    public:
    TwoSquareDetector () = default;

    std::pair<bool, std::pair<Pos, Pos>>
    validate_select (Player player, Piece piece, const Pos& pos) const;

    bool validate_move (Player player, Piece piece, const Pos& from_pos, const Pos& to_pos) const;

    void update (Player player, Piece piece, const Pos& from_pos, const Pos& to_pos);

    void reset () {
        p1_moves_.clear ();
        p2_moves_.clear ();
    }

    const std::vector<std::pair<Pos, Pos>>& get_player (Player player) const {
        return (player == Player::RED) ? p1_moves_ : p2_moves_;
    }
};

#endif // DETECTORS_H
