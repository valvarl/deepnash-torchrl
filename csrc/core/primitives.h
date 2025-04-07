#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>


enum class Piece : int8_t {
    EMPTY      = 0,
    LAKE       = 1,
    FLAG       = 2,
    BOMB       = 3,
    SPY        = 4,
    SCOUT      = 5,
    MINER      = 6,
    SERGEANT   = 7,
    LIEUTENANT = 8,
    CAPTAIN    = 9,
    MAJOR      = 10,
    COLONEL    = 11,
    GENERAL    = 12,
    MARSHAL    = 13
};

inline bool operator< (Piece lhs, Piece rhs) {
    return static_cast<int8_t> (lhs) < static_cast<int8_t> (rhs);
}

enum class Player : int8_t { RED = 1, BLUE = -1 };

using Pos = std::array<int8_t, 2>;

enum class GameMode { ORIGINAL, BARRAGE };

enum class GamePhase { TERMINAL, DEPLOY, SELECT, MOVE };

#endif // PRIMITIVES_H