#ifndef LPP_RANDOM_GEN_H
#define LPP_RANDOM_GEN_H

#include <random>

/*
Create probability distributions:

auto n = getNormalDistribution(4, 4.2);
auto u = getUniformDistribution(2.6, 23);
auto t = getStudentTDistribution(6);

Sample from them:

double n_val = n->sample();
double u_val = u->sample();
double t_val = t->sample();

For convience, we already have:

LPP::StandardNormalDistribution
LPP::StandardUniformDistribution

The distributions from STL will give doubles by default, but these can be converted to floats if desired.
If using custom numeric types, then you will have to write your own conversion from float/double.
*/

namespace LPP
{

class ProbabilityDistribution {
protected:
    static std::mt19937 MT_ENGINE;

public:
    virtual double sample() = 0; // Sample method will alter the state of the Mersenne Twister
};

class Normal : public ProbabilityDistribution {
    std::normal_distribution<> distnInstance;
    Normal(double mean, double stddev); // Private constructor
    friend std::shared_ptr<Normal> getNormalDistribution(double mean, double stddev);

public:
    ~Normal() {}
    double sample() override;
};

class Uniform : public ProbabilityDistribution {
    std::uniform_real_distribution<> distnInstance;
    Uniform(double lower, double upper); // Private constructor
    friend std::shared_ptr<Uniform> getUniformDistribution(double lowerBound, double upperBound);

public:
    ~Uniform() {}
    double sample() override;
};

class StudentT : public ProbabilityDistribution {
    std::student_t_distribution<> distnInstance;
    StudentT(unsigned int dof); // Private constructor
    friend std::shared_ptr<StudentT> getStudentTDistribution(unsigned int dof);

public:
    ~StudentT() {}
    double sample() override;
};

std::shared_ptr<Normal> getNormalDistribution(double mean, double stddev);
std::shared_ptr<Uniform> getUniformDistribution(double lowerBound, double upperBound);
std::shared_ptr<StudentT> getStudentTDistribution(unsigned int dof);

auto StandardNormalDistribution  = LPP::getNormalDistribution(0.0, 1.0);
auto StandardUniformDistribution = LPP::getUniformDistribution(0.0, 1.0);

} // namespace LPP

#endif