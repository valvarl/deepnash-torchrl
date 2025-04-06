#include "player_state_handler.h"
#include <algorithm>

PlayerStateHandler::PlayerStateHandler (Player player, size_t height, size_t width)
: player_ (player), deploy_idx_ (0), last_selected_ ({ -1, -1 }),
  last_selected_piece_ (Piece::EMPTY) {
}

void PlayerStateHandler::generate_state (const std::vector<int>& pieces_num,
const std::vector<bool>& deploy_mask,
size_t observed_history_entries,
size_t height,
size_t width) {

    // Reset state
    pieces_.clear ();
    movable_pieces_.clear ();
    deploy_idx_      = 0;
    deploy_mask_     = deploy_mask;
    public_obs_info_ = { std::vector<bool> (height * width, false),
        std::vector<bool> (height * width, false),
        std::vector<bool> (height * width, false) };
    observed_moves_.clear ();
    last_selected_       = { -1, -1 };
    last_selected_piece_ = Piece::EMPTY;

    // Initialize pieces and movable pieces
    for (size_t i = 0; i < pieces_num.size (); ++i) {
        if (pieces_num[i] > 0) {
            pieces_.push_back (static_cast<int> (i));

            // All pieces except FLAG and BOMB are movable
            if (i != static_cast<size_t> (Piece::FLAG) &&
            i != static_cast<size_t> (Piece::BOMB)) {
                movable_pieces_.push_back (static_cast<int> (i));
            }
        }
    }

    // Initialize unrevealed pieces counts
    unrevealed_ = pieces_num;

    // Initialize observed moves history
    observed_moves_.resize (observed_history_entries * height * width);
}