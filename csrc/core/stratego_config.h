#ifndef STRATEGO_CONFIG_H
#define STRATEGO_CONFIG_H

#include "primitives.h"
#include <unordered_map>
#include <vector>
#include <array>
#include <memory>

class StrategoEnv;

class StrategoConfig {
friend class StrategoEnv;
private:
    size_t height_;
    size_t width_;
    
    std::unordered_map<Piece, int> p1_pieces_;
    std::unordered_map<Piece, int> p2_pieces_;
    
    std::vector<bool> lakes_mask_;
    std::vector<bool> p1_deploy_mask_;
    std::vector<bool> p2_deploy_mask_;
    
    int total_moves_limit_;
    int moves_since_attack_limit_;
    int observed_history_entries_;
    bool allow_competitive_deploy_;
    GameMode game_mode_;

    std::vector<int> allowed_pieces_;

    static const std::unordered_map<Piece, int> PIECES_NUM_ORIGINAL;
    static const std::unordered_map<Piece, int> PIECES_NUM_BARRAGE;
    static const std::vector<std::pair<Pos, Pos>> PLACES_TO_DEPLOY_RED_ORIGINAL;
    static const std::vector<std::pair<Pos, Pos>> PLACES_TO_DEPLOY_BLUE_ORIGINAL;
    static const std::vector<std::pair<Pos, Pos>> LAKES_ORIGINAL;

    std::vector<bool> make_mask(const std::vector<std::pair<Pos, Pos>>& positions) const;
    std::pair<bool, std::string> validate() const;

public:
    StrategoConfig(
        size_t height,
        size_t width,
        const std::unordered_map<Piece, int>& p1_pieces,
        const std::unordered_map<Piece, int>& p2_pieces = {},
        const std::vector<std::pair<Pos, Pos>>& lakes = {},
        const std::vector<std::pair<Pos, Pos>>& p1_places_to_deploy = {},
        const std::vector<std::pair<Pos, Pos>>& p2_places_to_deploy = {},
        const std::vector<bool>& lakes_mask = {},
        const std::vector<bool>& p1_deploy_mask = {},
        const std::vector<bool>& p2_deploy_mask = {},
        int total_moves_limit = 2000,
        int moves_since_attack_limit = 200,
        int observed_history_entries = 40,
        bool allow_competitive_deploy = false,
        GameMode game_mode = GameMode::ORIGINAL
    );

    static StrategoConfig from_game_mode(GameMode mode);
    
    // Getters
    size_t height() const { return height_; }
    size_t width() const { return width_; }
    size_t total_moves_limit() const { return total_moves_limit_; }
    size_t moves_since_attack_limit() const { return moves_since_attack_limit_; }
    size_t observed_history_entries() const { return observed_history_entries_; }
    const std::vector<int>& allowed_pieces() const { return allowed_pieces_; }
    const std::vector<bool>& lakes_mask() const { return lakes_mask_; }
    const std::vector<bool>& p1_deploy_mask() const { return p1_deploy_mask_; }
    const std::vector<bool>& p2_deploy_mask() const { return p2_deploy_mask_; }
    const std::unordered_map<Piece, int>& p1_pieces() const { return p1_pieces_; }
    const std::unordered_map<Piece, int>& p2_pieces() const { return p2_pieces_; }
    GameMode game_mode() const { return game_mode_; }
};

#endif // STRATEGO_CONFIG_H
