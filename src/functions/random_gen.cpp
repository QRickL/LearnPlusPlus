#include "random_gen.h"
#include "../check/check.h"

std::mt19937 LPP::ProbabilityDistribution::MT_ENGINE = std::mt19937(std::random_device{}());

//LPP::ProbabilityDistribution::~ProbabilityDistribution() {}

LPP::Normal::Normal(float mean, float stddev) {
    distnInstance = std::normal_distribution<>(mean, stddev);
}

float LPP::Normal::sample() {
    return distnInstance(ProbabilityDistribution::MT_ENGINE);
}

LPP::Uniform::Uniform(float lower, float upper) {
    distnInstance = std::uniform_real_distribution<>(lower, upper);
}

float LPP::Uniform::sample() {
    return distnInstance(ProbabilityDistribution::MT_ENGINE);
}

LPP::StudentT::StudentT(unsigned int dof) {
    distnInstance = std::student_t_distribution<>(dof);
}

float LPP::StudentT::sample() {
    return distnInstance(ProbabilityDistribution::MT_ENGINE);
}

std::shared_ptr<LPP::Normal> LPP::getNormalDistribution(float mean, float stddev)
{
    _lpp_check_(stddev > 0.0, "Normal: standard deviation must be positive");
    return std::shared_ptr<LPP::Normal>(new LPP::Normal(mean, stddev));
}

std::shared_ptr<LPP::Uniform> LPP::getUniformDistribution(float lowerBound, float upperBound)
{
    _lpp_check_(upperBound > lowerBound, "Uniform: lower cannot be greater than higher");
    return std::shared_ptr<LPP::Uniform>(new LPP::Uniform(upperBound, upperBound));
}

std::shared_ptr<LPP::StudentT> LPP::getStudentTDistribution(unsigned int dof)
{
    return std::shared_ptr<LPP::StudentT>(new LPP::StudentT(dof));
}