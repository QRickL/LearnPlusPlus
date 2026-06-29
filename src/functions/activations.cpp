#include "activations.h"
#include <cmath>

LPP::Activation::~Activation() {}

double LPP::Identity::apply_activation(double x) const
{
    return x;
}

double LPP::Identity::apply_derivative(double x) const
{
    return 1.0;
}

const std::string& LPP::Identity::who() const {
    return STRING_IDENTITY;
}

double LPP::ReLU::apply_activation(double x) const
{
    return (x >= 0) ? x : 0.0;
}

double LPP::ReLU::apply_derivative(double x) const 
{
    // Mathematically speaking, it should be undefined at zero
    // For convenience, we assign it 1
    return (x >= 0) ? 1 : 0.0;
}

const std::string& LPP::ReLU::who() const {
    return STRING_RELU;
}

double LPP::Sigmoid::apply_activation(double x) const
{
    return 1 / (1 + std::exp(-x));
}

double LPP::Sigmoid::apply_derivative(double x) const
{
    const double s = LPP::Sigmoid::apply_activation(x);
    return s * (1-s);
}

const std::string& LPP::Sigmoid::who() const {
    return STRING_SIGMOID;
}

double LPP::Tanh::apply_activation(double x) const
{
    return std::tanh(x);
}

double LPP::Tanh::apply_derivative(double x) const
{
    const double s = std::cosh(x);
    return 1 / (s * s);
}

const std::string& LPP::Tanh::who() const {
    return STRING_TANH;
}
