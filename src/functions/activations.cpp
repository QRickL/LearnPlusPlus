#include "activations.h"
#include <cmath>

template <typename number>
LPP::Activation<number>::~Activation() {}

template <typename number>
number LPP::Identity<number>::applyActivation(number x) const
{
    return x;
}

template <typename number>
number LPP::Identity<number>::applyDerivative(number x) const
{
    return 1.0;
}

template <typename number>
const std::string& LPP::Identity<number>::who() const {
    return STRING_IDENTITY;
}

template <typename number>
number LPP::ReLU<number>::applyActivation(number x) const
{
    return (x >= 0.0) ? x : 0.0;
}

template <typename number>
number LPP::ReLU<number>::applyDerivative(number x) const 
{
    // Mathematically speaking, it should be undefined at zero
    // For convenience, we assign it 1
    return (x >= 0.0) ? 1 : 0.0;
}

template <typename number>
const std::string& LPP::ReLU<number>::who() const {
    return STRING_RELU;
}

template <typename number>
number LPP::Sigmoid<number>::applyActivation(number x) const
{
    return 1.0 / (1.0 + std::exp(-x));
}

template <typename number>
number LPP::Sigmoid<number>::applyDerivative(number x) const
{
    const number s = LPP::Sigmoid::applyActivation(x);
    return s * (1.0-s);
}

template <typename number>
const std::string& LPP::Sigmoid<number>::who() const {
    return STRING_SIGMOID;
}

template <typename number>
number LPP::Tanh<number>::applyActivation(number x) const
{
    return std::tanh(x);
}

template <typename number>
number LPP::Tanh<number>::applyDerivative(number x) const
{
    const number s = std::cosh(x);
    return 1.0 / (s * s);
}

template <typename number>
const std::string& LPP::Tanh<number>::who() const {
    return STRING_TANH;
}
