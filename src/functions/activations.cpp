#include "activations.h"
#include <cmath>

LPP::Activation::~Activation() {}

float LPP::Identity::apply_activation(float x) const
{
    return x;
}

float LPP::Identity::apply_derivative(float x) const
{
    return 1.0;
}

const std::string& LPP::Identity::who() const {
    return STRING_IDENTITY;
}

float LPP::ReLU::apply_activation(float x) const
{
    return (x >= 0) ? x : 0.0;
}

float LPP::ReLU::apply_derivative(float x) const 
{
    // Mathematically speaking, it should be undefined at zero
    // For convenience, we assign it 1
    return (x >= 0) ? 1 : 0.0;
}

const std::string& LPP::ReLU::who() const {
    return STRING_RELU;
}

float LPP::Sigmoid::apply_activation(float x) const
{
    return 1 / (1 + std::exp(-x));
}

float LPP::Sigmoid::apply_derivative(float x) const
{
    const float s = LPP::Sigmoid::apply_activation(x);
    return s * (1-s);
}

const std::string& LPP::Sigmoid::who() const {
    return STRING_SIGMOID;
}

float LPP::Tanh::apply_activation(float x) const
{
    return std::tanh(x);
}

float LPP::Tanh::apply_derivative(float x) const
{
    const float s = std::cosh(x);
    return 1 / (s * s);
}

const std::string& LPP::Tanh::who() const {
    return STRING_TANH;
}
