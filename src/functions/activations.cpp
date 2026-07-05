#include "activations.hpp"
#include <cmath>

// No need to inline. Compiler will do it for you

LPP::activations::Activation::~Activation() {}

float LPP::activations::Identity::apply_activation(float x) const
{
    return x;
}

float LPP::activations::Identity::apply_derivative(float x) const
{
    return 1.f;
}

const std::string& LPP::activations::Identity::who() const {
    return STRING_IDENTITY_;
}

float LPP::activations::ReLU::apply_activation(float x) const
{
    return (x >= 0.f) ? x : 0.f;
}

float LPP::activations::ReLU::apply_derivative(float x) const 
{
    // Mathematically speaking, it should be undefined at zero
    // For convenience, we assign it 1
    return (x >= 0.f) ? 1.f : 0.f;
}

const std::string& LPP::activations::ReLU::who() const {
    return STRING_RELU_;
}

float LPP::activations::Sigmoid::apply_activation(float x) const
{
    return 1.f / (1.f + std::exp(-x));
}

float LPP::activations::Sigmoid::apply_derivative(float x) const
{
    float s = LPP::activations::Sigmoid::apply_activation(x);
    return s * (1.f-s);
}

const std::string& LPP::activations::Sigmoid::who() const {
    return STRING_SIGMOID_;
}

float LPP::activations::Tanh::apply_activation(float x) const
{
    return std::tanh(x);
}

float LPP::activations::Tanh::apply_derivative(float x) const
{
    float s = std::cosh(x);
    return 1.f / (s * s);
}

const std::string& LPP::activations::Tanh::who() const {
    return STRING_TANH_;
}
