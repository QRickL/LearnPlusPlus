#include "random_gen.h"

std::mt19937 LPP::ProbabilityDistribution::MT_ENGINE = std::mt19937(std::random_device{}());

LPP::ProbabilityDistribution::~ProbabilityDistribution() {}

LPP::Normal::Normal(const float mean, const float stddev)
{
    if (stddev < 0) {
        const auto msg = "Normal: standard deviation cannot be negative";
        throw std::invalid_argument(msg);
    }
    norm = std::normal_distribution<>(mean, stddev);
}

float LPP::Normal::sample()
{
    return norm(ProbabilityDistribution::MT_ENGINE);
}

LPP::Uniform::Uniform(const float lower, const float upper)
{
    if (lower > upper) {
        const auto msg = "Uniform: lower cannot be greater than higher";
        throw std::invalid_argument(msg);
    }
    unif = std::uniform_real_distribution<>(lower, upper);
}

float LPP::Uniform::sample()
{
    return unif(ProbabilityDistribution::MT_ENGINE);
}