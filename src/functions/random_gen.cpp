#include "random_gen.h"
#include "../checking/check.h"

std::mt19937 LPP::ProbabilityDistribution::MT_ENGINE = std::mt19937(std::random_device{}());

LPP::ProbabilityDistribution::~ProbabilityDistribution() {}

LPP::Normal::Normal(float mean, float stddev)
{
    __lpp_check__(stddev > 0.f, "Normal::Normal - stddev must be positive");

    norm = std::normal_distribution<>(mean, stddev);
}

float LPP::Normal::sample()
{
    return norm(ProbabilityDistribution::MT_ENGINE);
}

LPP::Uniform::Uniform(float lower, float upper)
{
    __lpp_check__(upper > lower, "Uniform::Uniform - must have upper > lower");
    
    unif = std::uniform_real_distribution<>(lower, upper);
}

float LPP::Uniform::sample()
{
    return unif(ProbabilityDistribution::MT_ENGINE);
}