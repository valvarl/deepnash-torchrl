#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include <array>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <stdexcept>
#include <memory>

enum class Piece : int8_t {
    EMPTY = 0,
    LAKE = 1,
    FLAG = 2,
    BOMB = 3,
    SPY = 4,
    SCOUT = 5,
    MINER = 6,
    SERGEANT = 7,
    LIEUTENANT = 8,
    CAPTAIN = 9,
    MAJOR = 10,
    COLONEL = 11,
    GENERAL = 12,
    MARSHAL = 13
};

inline bool operator<(Piece lhs, Piece rhs) {
    return static_cast<int8_t>(lhs) < static_cast<int8_t>(rhs);
}

enum class Player : int8_t {
    RED = 1,
    BLUE = -1
};

using Pos = std::array<int8_t, 2>;

enum class GameMode {
    ORIGINAL,
    BARRAGE
};

enum class GamePhase {
    TERMINAL,
    DEPLOY,
    SELECT,
    MOVE
};

class StrategoEnv;

template<typename T>
class Matrix {
friend class StrategoEnv;
private:
    std::vector<T> data_;
    size_t height_;
    size_t width_;

public:
    Matrix(size_t height, size_t width, T init_val = T())
        : height_(height), width_(width), data_(height * width, init_val) {}

    // Для не-bool типов
    T& operator()(size_t row, size_t col) { return data_[row * width_ + col]; }
    const T& operator()(size_t row, size_t col) const { return data_[row * width_ + col]; }

    size_t height() const { return height_; }
    size_t width() const { return width_; }
    void fill(T value) { std::fill(data_.begin(), data_.end(), value); }
};

// Специализация для bool
template<>
class Matrix<bool> {
friend class StrategoEnv;
private:
    std::vector<bool> data_;
    size_t height_;
    size_t width_;

public:
    Matrix(size_t height, size_t width, bool init_val = false)
        : height_(height), width_(width), data_(height * width, init_val) {}

    // Специальные методы для bool
    std::vector<bool>::reference operator()(size_t row, size_t col) {
        return data_[row * width_ + col];
    }
    
    bool operator()(size_t row, size_t col) const {
        return data_[row * width_ + col];
    }

    const std::vector<bool>& data() const { return data_; }
    size_t height() const { return height_; }
    size_t width() const { return width_; }

    void fill(bool value) { std::fill(data_.begin(), data_.end(), value); }
};

#endif  // PRIMITIVES_H