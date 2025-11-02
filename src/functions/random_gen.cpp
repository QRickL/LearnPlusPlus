#include "random_gen.h"

std::mt19937 LPP::ProbabilityDistribution::MT_ENGINE = std::mt19937(std::random_device{}());

LPP::ProbabilityDistribution::~ProbabilityDistribution() {}

LPP::Normal::Normal(const double mean, const double stddev)
{
    if (stddev < 0) {
        const auto msg = "Normal: standard deviation cannot be negative";
        throw std::invalid_argument(msg);
    }
    norm = std::normal_distribution<>(mean, stddev);
}

double LPP::Normal::sample()
{
    return norm(ProbabilityDistribution::MT_ENGINE);
}

LPP::Uniform::Uniform(const double lower, const double upper)
{
    if (lower > upper) {
        const auto msg = "Uniform: lower cannot be greater than higher";
        throw std::invalid_argument(msg);
    }
    unif = std::uniform_real_distribution<>(lower, upper);
}

double LPP::Uniform::sample()
{
    return unif(ProbabilityDistribution::MT_ENGINE);
}