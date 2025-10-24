#include "activations.h"
#include <cmath>

LPP::Activation::~Activation() {}

// Identity
double LPP::Identity::apply_itself(double x) const
{
    return x;
}

double LPP::Identity::apply_derivative(double x) const
{
    return 1.0;
}

std::string LPP::Identity::who() const {
    return STRING_IDENTITY;
}

// ReLU (rectified linear unit)
double LPP::ReLU::apply_itself(double x) const
{
    return (x >= 0) ? x : 0.0;
}

double LPP::ReLU::apply_derivative(double x) const 
{
    // Mathematically speaking, it should be undefined at zero
    // For convenience, we assign it 1
    return (x >= 0) ? 1 : 0.0;
}

std::string LPP::ReLU::who() const {
    return STRING_RELU;
}

// Sigmoid
double LPP::Sigmoid::apply_itself(double x) const
{
    return 1 / (1 + std::exp(-x));
}

double LPP::Sigmoid::apply_derivative(double x) const
{
    const double s = LPP::Sigmoid::apply_itself(x);
    return s * (1-s);
}

std::string LPP::Sigmoid::who() const {
    return STRING_SIGMOID;
}

// Tanh
double LPP::Tanh::apply_itself(double x) const
{
    return std::tanh(x);
}

double LPP::Tanh::apply_derivative(double x) const
{
    const double s = std::cosh(x);
    return 1 / (s * s);
}

std::string LPP::Tanh::who() const {
    return STRING_TANH;
}
