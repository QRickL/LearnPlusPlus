#ifndef LPP_RANDOM_GEN_HPP
#define LPP_RANDOM_GEN_HPP

#include <random>

/*
Note: distributions should not be marked const

One thing I learned that is very important is the variance of the distribution we draw the weights from
*/

namespace LPP {
namespace distribution {

class ProbabilityDistribution {
protected:
    static std::mt19937 mt_engine_;

public:
    virtual ~ProbabilityDistribution() = 0;

    virtual float sample() = 0;
};

class Normal : public ProbabilityDistribution {
    std::normal_distribution<float> norm;

public:
    Normal(float mean, float stddev);
    ~Normal() {}

    float sample() override;
};

class Uniform : public ProbabilityDistribution {
    std::uniform_real_distribution<float> unif;

public:
    Uniform(float lower, float upper);
    ~Uniform() {}

    float sample() override;
};

inline Normal standard_normal = Normal(0, 1);
inline Uniform standard_uniform = Uniform(0, 1);

} // namespace distribution
} // namespace LPP

#endif