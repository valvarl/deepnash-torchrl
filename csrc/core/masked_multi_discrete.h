#ifndef MASKED_MULTI_DISCRETE_H
#define MASKED_MULTI_DISCRETE_H

#include <vector>
#include <random>
#include <algorithm>

class MaskedMultiDiscrete {
private:
    std::vector<int> nvec_;
    std::vector<bool> mask_;
    std::mt19937 rng_;

public:
    // Объявление конструктора
    MaskedMultiDiscrete(const std::vector<int>& nvec, uint32_t seed = 0);
    
    // Объявление методов
    void set_mask(const std::vector<bool>& mask);
    std::vector<int> sample();
};

#endif  // MASKED_MULTI_DISCRETE_H