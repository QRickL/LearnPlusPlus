#ifndef LPP_RANDOM_GEN_H
#define LPP_RANDOM_GEN_H

#include <random>

namespace LPP
{

class ProbabilityDistribution {
protected:
    static std::mt19937 MT_ENGINE;

public:
    virtual ~ProbabilityDistribution() = 0;

    virtual float sample() = 0;
};

class Normal : public ProbabilityDistribution {
    std::normal_distribution<> norm;

public:
    Normal(float mean, float stddev);
    ~Normal() {}

    float sample() override;
};

class Uniform : public ProbabilityDistribution {
    std::uniform_real_distribution<> unif;

public:
    Uniform(float lower, float upper);
    ~Uniform() {}

    float sample() override;
};

const auto STANDARD_NORMAL = std::make_shared<Normal>(0, 1);
const auto CENTRAL_UNIF = std::make_shared<Uniform>(-1, 1);

} // namespace LPP

#endif