#ifndef LPP_RANDOM_GEN_H
#define LPP_RANDOM_GEN_H

#include <random>

/*
Create probability distributions:

auto n = getNormalDistribution(4, 4.2);
auto u = getUniformDistribution(2.6, 23);
auto t = getStudentTDistribution(6);

Sample from them:

float n_val = n->sample();
float u_val = u->sample();
float t_val = t->sample();

For convience, we already have:

LPP::StandardNormalDistribution
LPP::StandardUniformDistribution

If using custom numeric types, then you will have to write your own conversion from float/float.
*/

namespace LPP
{

class ProbabilityDistribution {
protected:
    static std::mt19937 MT_ENGINE;

public:
    virtual float sample() = 0; // Sample method will alter the state of the Mersenne Twister
};

class Normal : public ProbabilityDistribution {
    std::normal_distribution<> distnInstance;
    Normal(float mean, float stddev); // Private constructor
    friend std::shared_ptr<Normal> getNormalDistribution(float mean, float stddev);

public:
    ~Normal() {}
    float sample() override;
};

class Uniform : public ProbabilityDistribution {
    std::uniform_real_distribution<> distnInstance;
    Uniform(float lower, float upper); // Private constructor
    friend std::shared_ptr<Uniform> getUniformDistribution(float lowerBound, float upperBound);

public:
    ~Uniform() {}
    float sample() override;
};

class StudentT : public ProbabilityDistribution {
    std::student_t_distribution<> distnInstance;
    StudentT(unsigned int dof); // Private constructor
    friend std::shared_ptr<StudentT> getStudentTDistribution(unsigned int dof);

public:
    ~StudentT() {}
    float sample() override;
};

std::shared_ptr<Normal> getNormalDistribution(float mean, float stddev);
std::shared_ptr<Uniform> getUniformDistribution(float lowerBound, float upperBound);
std::shared_ptr<StudentT> getStudentTDistribution(unsigned int dof);

auto StandardNormalDistribution  = LPP::getNormalDistribution(0.0, 1.0);
auto StandardUniformDistribution = LPP::getUniformDistribution(0.0, 1.0);

} // namespace LPP

#endif