#include "stratego_config.h"
#include <algorithm>
#include <stdexcept>

// Static member initialization
const std::unordered_map<Piece, int> StrategoConfig::PIECES_NUM_ORIGINAL = {
    {Piece::FLAG, 1}, {Piece::BOMB, 6}, {Piece::SPY, 1}, {Piece::SCOUT, 8},
    {Piece::MINER, 5}, {Piece::SERGEANT, 4}, {Piece::LIEUTENANT, 4},
    {Piece::CAPTAIN, 4}, {Piece::MAJOR, 3}, {Piece::COLONEL, 2},
    {Piece::GENERAL, 1}, {Piece::MARSHAL, 1}
};

const std::unordered_map<Piece, int> StrategoConfig::PIECES_NUM_BARRAGE = {
    {Piece::FLAG, 1}, {Piece::BOMB, 1}, {Piece::SPY, 1}, {Piece::SCOUT, 2},
    {Piece::MINER, 1}, {Piece::GENERAL, 1}, {Piece::MARSHAL, 1}
};

const std::vector<std::pair<Pos, Pos>> StrategoConfig::PLACES_TO_DEPLOY_RED_ORIGINAL = {
    {{Pos{6, 0}, Pos{9, 9}}}
};

const std::vector<std::pair<Pos, Pos>> StrategoConfig::PLACES_TO_DEPLOY_BLUE_ORIGINAL = {
    {{Pos{0, 0}, Pos{3, 9}}}
};

const std::vector<std::pair<Pos, Pos>> StrategoConfig::LAKES_ORIGINAL = {
    {std::make_pair<Pos, Pos>(Pos{4, 2}, Pos{5, 3})}, 
    {std::make_pair<Pos, Pos>(Pos{4, 6}, Pos{5, 7})},
};

Matrix<bool> StrategoConfig::make_mask(const std::vector<std::pair<Pos, Pos>>& positions) const {
    Matrix<bool> mask(height_, width_, false);
    
    for (const auto& [top_left, bottom_right] : positions) {
        int y1 = std::min(top_left[0], bottom_right[0]);
        int y2 = std::max(top_left[0], bottom_right[0]);
        int x1 = std::min(top_left[1], bottom_right[1]);
        int x2 = std::max(top_left[1], bottom_right[1]);
        
        for (int y = y1; y <= y2; ++y) {
            for (int x = x1; x <= x2; ++x) {
                mask(y, x) = true;
            }
        }
    }
    
    return mask;
}

std::pair<bool, std::string> StrategoConfig::validate() const {
    // Check deployment overlaps with lakes
    for (size_t y = 0; y < height_; ++y) {
        for (size_t x = 0; x < width_; ++x) {
            if (p1_deploy_mask_(y, x) && lakes_mask_(y, x)) {
                return {false, "Player 1's deployment overlaps with lakes"};
            }
            if (p2_deploy_mask_(y, x) && lakes_mask_(y, x)) {
                return {false, "Player 2's deployment overlaps with lakes"};
            }
        }
    }

    if (!allow_competitive_deploy_) {
        // Check deployment overlaps between players
        for (size_t y = 0; y < height_; ++y) {
            for (size_t x = 0; x < width_; ++x) {
                if (p1_deploy_mask_(y, x) && p2_deploy_mask_(y, x)) {
                    return {false, "Player 1's and Player 2's deployments overlap"};
                }
            }
        }

        // Check enough deployment spots
        int p1_spots = 0, p2_spots = 0;
        int p1_pieces = 0, p2_pieces = 0;
        
        for (size_t y = 0; y < height_; ++y) {
            for (size_t x = 0; x < width_; ++x) {
                p1_spots += p1_deploy_mask_(y, x);
                p2_spots += p2_deploy_mask_(y, x);
            }
        }
        
        for (const auto& [piece, count] : p1_pieces_) {
            p1_pieces += count;
        }
        for (const auto& [piece, count] : p2_pieces_) {
            p2_pieces += count;
        }
        
        if (p1_spots < p1_pieces) {
            return {false, "Player 1 has fewer deployment spots than pieces"};
        }
        if (p2_spots < p2_pieces) {
            return {false, "Player 2 has fewer deployment spots than pieces"};
        }
    } else {
        // Competitive deployment validation
        int p1_own = 0, p2_own = 0;
        int shared = 0;
        
        for (size_t y = 0; y < height_; ++y) {
            for (size_t x = 0; x < width_; ++x) {
                bool p1 = p1_deploy_mask_(y, x);
                bool p2 = p2_deploy_mask_(y, x);
                
                if (p1 && !p2) p1_own++;
                if (p2 && !p1) p2_own++;
                if (p1 && p2) shared++;
            }
        }
        
        int p1_pieces = 0, p2_pieces = 0;
        for (const auto& [piece, count] : p1_pieces_) {
            p1_pieces += count;
        }
        for (const auto& [piece, count] : p2_pieces_) {
            p2_pieces += count;
        }
        
        if ((p1_pieces - p1_own) + (p2_pieces - p2_own) > shared) {
            return {false, "Total number of pieces exceeds available shared deployment spots"};
        }
    }
    
    if (total_moves_limit_ <= 0) {
        return {false, "Total moves limit must be greater than 0"};
    }
    
    if (moves_since_attack_limit_ <= 0) {
        return {false, "Moves since last attack limit must be greater than 0"};
    }
    
    if (observed_history_entries_ < 0) {
        return {false, "Observed history entries cannot be negative"};
    }
    
    return {true, "Validation successful"};
}

StrategoConfig::StrategoConfig(
    size_t height,
    size_t width,
    const std::unordered_map<Piece, int>& p1_pieces,
    const std::unordered_map<Piece, int>& p2_pieces,
    const std::vector<std::pair<Pos, Pos>>& lakes,
    const std::vector<std::pair<Pos, Pos>>& p1_places_to_deploy,
    const std::vector<std::pair<Pos, Pos>>& p2_places_to_deploy,
    const Matrix<bool>& lakes_mask,
    const Matrix<bool>& p1_deploy_mask,
    const Matrix<bool>& p2_deploy_mask,
    int total_moves_limit,
    int moves_since_attack_limit,
    int observed_history_entries,
    bool allow_competitive_deploy,
    GameMode game_mode
) : height_(height), width_(width),
    p1_pieces_(p1_pieces),
    p2_pieces_(p2_pieces.empty() ? p1_pieces : p2_pieces),
    lakes_mask_(lakes_mask.height() == 0 ? make_mask(lakes) : lakes_mask),
    p1_deploy_mask_(p1_deploy_mask.height() == 0 ? make_mask(p1_places_to_deploy) : p1_deploy_mask),
    p2_deploy_mask_(p2_deploy_mask.height() == 0 ? make_mask(p2_places_to_deploy) : p2_deploy_mask),
    total_moves_limit_(total_moves_limit),
    moves_since_attack_limit_(moves_since_attack_limit),
    observed_history_entries_(observed_history_entries),
    allow_competitive_deploy_(allow_competitive_deploy),
    game_mode_(game_mode) {
    
    allowed_pieces_.clear();
    for (size_t i = 0; i < p1_pieces_.size(); ++i) {
        Piece piece = static_cast<Piece>(i);
        int p1_count = 0, p2_count = 0;
    
        if (p1_pieces_.count(piece)) {
            p1_count = p1_pieces_.at(piece);
        }
        if (p2_pieces_.count(piece)) {
            p2_count = p2_pieces_.at(piece);
        }
    
        if (p1_count != 0 || p2_count != 0) {
            allowed_pieces_.push_back(static_cast<int>(piece));
        }
    }

    auto [valid, msg] = validate();
    if (!valid) {
        throw std::invalid_argument(msg);
    }
}

StrategoConfig StrategoConfig::from_game_mode(GameMode mode) {
    switch (mode) {
        case GameMode::ORIGINAL:
            return StrategoConfig(
                10, 10,
                PIECES_NUM_ORIGINAL,
                {},
                LAKES_ORIGINAL,
                PLACES_TO_DEPLOY_RED_ORIGINAL,
                PLACES_TO_DEPLOY_BLUE_ORIGINAL,
                Matrix<bool>(0, 0), Matrix<bool>(0, 0), Matrix<bool>(0, 0),
                2000, 200, 40, false, GameMode::ORIGINAL
            );
        case GameMode::BARRAGE:
            return StrategoConfig(
                10, 10,
                PIECES_NUM_BARRAGE,
                {},
                LAKES_ORIGINAL,
                PLACES_TO_DEPLOY_RED_ORIGINAL,
                PLACES_TO_DEPLOY_BLUE_ORIGINAL,
                Matrix<bool>(0, 0), Matrix<bool>(0, 0), Matrix<bool>(0, 0),
                2000, 200, 40, false, GameMode::BARRAGE
            );
        default:
            throw std::invalid_argument("Unknown game mode");
    }
}

StrategoConfig StrategoConfig::rot90(int k) const {
    k = k % 4;
    if (k < 0) k += 4;
    
    size_t new_height = (k % 2 == 0) ? height_ : width_;
    size_t new_width = (k % 2 == 0) ? width_ : height_;
    
    auto rotate_mask = [k, new_height, new_width](const Matrix<bool>& mask) {
        Matrix<bool> rotated(new_height, new_width);
        for (size_t y = 0; y < mask.height(); ++y) {
            for (size_t x = 0; x < mask.width(); ++x) {
                size_t new_y, new_x;
                switch (k) {
                    case 1: // 90 degrees
                        new_y = x;
                        new_x = mask.height() - 1 - y;
                        break;
                    case 2: // 180 degrees
                        new_y = mask.height() - 1 - y;
                        new_x = mask.width() - 1 - x;
                        break;
                    case 3: // 270 degrees
                        new_y = mask.width() - 1 - x;
                        new_x = y;
                        break;
                    default: // 0 degrees
                        new_y = y;
                        new_x = x;
                        break;
                }
                rotated(new_y, new_x) = mask(y, x);
            }
        }
        return rotated;
    };
    
    return StrategoConfig(
        new_height, new_width,
        p1_pieces_,
        p2_pieces_,
        {}, // positions will be ignored since we provide masks
        {},
        {},
        rotate_mask(lakes_mask_),
        rotate_mask(p1_deploy_mask_),
        rotate_mask(p2_deploy_mask_),
        total_moves_limit_,
        moves_since_attack_limit_,
        observed_history_entries_,
        allow_competitive_deploy_,
        game_mode_
    );
}
