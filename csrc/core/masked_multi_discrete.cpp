#include "masked_multi_discrete.h"

#include <numeric>
#include <stdexcept>


// Реализация конструктора
MaskedMultiDiscrete::MaskedMultiDiscrete (const std::vector<int>& nvec, uint32_t seed)
: nvec_ (nvec),
  mask_ (std::accumulate (nvec.begin (), nvec.end (), 1, std::multiplies<int> ()), true) {
    if (seed == 0) {
        std::random_device rd;
        seed = rd ();
    }
    rng_.seed (seed);
}

// Реализация set_mask
void MaskedMultiDiscrete::set_mask (const std::vector<bool>& mask) {
    if (mask.size () != mask_.size ()) {
        throw std::invalid_argument ("Mask size must match space size");
    }
    mask_ = mask;
}

// Реализация sample
std::vector<int> MaskedMultiDiscrete::sample () {
    // If no mask (all true), sample from entire space
    if (std::all_of (mask_.begin (), mask_.end (), [] (bool v) { return v; })) {
        std::vector<int> result;
        for (int n : nvec_) {
            std::uniform_int_distribution<int> dist (0, n - 1);
            result.push_back (dist (rng_));
        }
        return result;
    }

    // Collect valid indices
    std::vector<size_t> valid_indices;
    for (size_t i = 0; i < mask_.size (); ++i) {
        if (mask_[i])
            valid_indices.push_back (i);
    }

    if (valid_indices.empty ()) {
        throw std::runtime_error ("No valid actions in the mask!");
    }

    // Randomly pick one valid index
    std::uniform_int_distribution<size_t> dist (0, valid_indices.size () - 1);
    size_t flat_idx = valid_indices[dist (rng_)];

    // Convert flat index to multi-dimensional coordinates
    std::vector<int> result;
    size_t remaining = flat_idx;
    for (int n : nvec_) {
        result.push_back (remaining % n);
        remaining /= n;
    }

    return result;
}
