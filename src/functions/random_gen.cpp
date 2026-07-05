#include "random_gen.hpp"
#include "../checking/check.hpp"

std::mt19937 LPP::distribution::ProbabilityDistribution::mt_engine_ = std::mt19937(std::random_device{}());

LPP::distribution::ProbabilityDistribution::~ProbabilityDistribution() {}

LPP::distribution::Normal::Normal(float mean, float stddev)
{
    __lpp_check__(stddev > 0.f, "Normal::Normal - stddev must be positive");

    norm = std::normal_distribution<>(mean, stddev);
}

float LPP::distribution::Normal::sample()
{
    return norm(ProbabilityDistribution::mt_engine_);
}

LPP::distribution::Uniform::Uniform(float lower, float upper)
{
    __lpp_check__(upper > lower, "Uniform::Uniform - must have upper > lower");
    
    unif = std::uniform_real_distribution<>(lower, upper);
}

float LPP::distribution::Uniform::sample()
{
    return unif(ProbabilityDistribution::mt_engine_);
}