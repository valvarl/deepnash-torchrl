#ifndef MASKED_MULTI_DISCRETE_H
#define MASKED_MULTI_DISCRETE_H

#include <algorithm>
#include <random>
#include <vector>


class MaskedMultiDiscrete {
    private:
    std::vector<int> nvec_;
    std::vector<bool> mask_;
    std::mt19937 rng_;

    public:
    // Объявление конструктора
    MaskedMultiDiscrete (const std::vector<int>& nvec, uint32_t seed = 0);

    // Объявление методов
    void set_mask (const std::vector<bool>& mask);
    std::vector<int> sample ();

    const std::vector<int>& nvec () const {
        return nvec_;
    }

    const std::vector<bool>& mask () const {
        return mask_;
    }
};

#endif // MASKED_MULTI_DISCRETE_H