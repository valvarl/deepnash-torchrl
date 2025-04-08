#ifndef MASKED_MULTI_DISCRETE_H
#define MASKED_MULTI_DISCRETE_H

#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>


class MaskedMultiDiscrete {
    private:
    std::vector<int> nvec_;
    std::vector<bool> mask_;
    std::mt19937 rng_;

    public:
    // Constructor
    MaskedMultiDiscrete (const std::vector<int>& nvec, uint32_t seed = 0);

    // Method to set the mask
    void set_mask (const std::vector<bool>& mask);

    // Method to sample from the space
    std::vector<int> sample ();

    // Getter for nvec
    const std::vector<int>& nvec () const {
        return nvec_;
    }

    // Getter for mask
    const std::vector<bool>& mask () const {
        return mask_;
    }

    void update_rng_state () {
        rng_.discard (1);
    }
};

#endif // MASKED_MULTI_DISCRETE_H