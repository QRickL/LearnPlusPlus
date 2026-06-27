#include "random_gen.h"
#include "../check/check.h"

std::mt19937 LPP::ProbabilityDistribution::MT_ENGINE = std::mt19937(std::random_device{}());

LPP::ProbabilityDistribution::~ProbabilityDistribution() {}

LPP::Normal::Normal(double mean, double stddev) {
    distnInstance = std::normal_distribution<>(mean, stddev);
}

double LPP::Normal::sample() {
    return distnInstance(ProbabilityDistribution::MT_ENGINE);
}

LPP::Uniform::Uniform(double lower, double upper) {
    distnInstance = std::uniform_real_distribution<>(lower, upper);
}

double LPP::Uniform::sample() {
    return distnInstance(ProbabilityDistribution::MT_ENGINE);
}

LPP::StudentT::StudentT(unsigned int dof) {
    distnInstance = std::student_t_distribution<>(dof);
}

double LPP::StudentT::sample() {
    return distnInstance(ProbabilityDistribution::MT_ENGINE);
}

std::shared_ptr<LPP::Normal> getNormalDistribution(double mean, double stddev)
{
    LPP::_check_(stddev > 0.0, "Normal: standard deviation must be positive");
    return std::make_shared<LPP::Normal>();
}

std::shared_ptr<LPP::Uniform> getUniformDistribution(double lowerBound, double upperBound)
{
    LPP::_check_(upperBound > lowerBound, "Uniform: lower cannot be greater than higher");
    return std::make_shared<LPP::Uniform>();
}

std::shared_ptr<LPP::StudentT> getStudentTDistribution(unsigned int dof)
{
    return std::make_shared<LPP::StudentT>(dof);
}