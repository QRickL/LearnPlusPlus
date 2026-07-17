#include "activations.hpp"
#include "../checking/check.hpp"
#include <cmath>

// No need to inline. Compiler will do it for you

LPP::activations::Activation::~Activation() {}

void LPP::activations::Activation::apply_activation(std::vector<float>& v) const
{
    for (float& x : v) x = apply_activation_(x);
}

float LPP::activations::Identity::apply_activation_(float x) const
{
    return x;
}

float LPP::activations::Identity::apply_derivative(float x) const
{
    return 1.f;
}

void LPP::activations::Activation::calculate_jacobian(const std::vector<float>& fv, LPP::Matrix& J) const
{
    __lpp_check__(J.rows() == J.cols(),
        "Activation::calculate_jacobian - given matrix is non-square");
    __lpp_check__(fv.size() == J.rows(),
        "Activation::calculate_jacobian - vector and matrix sizes disagree");
    enforce_condition(false,
        "Activation::calculate_jacobian - TODO: calculate_jacobian has not yet been implemented for this loss function :(");
}


const std::string& LPP::activations::Identity::who() const {
    return STRING_IDENTITY_;
}

float LPP::activations::ReLU::apply_activation_(float x) const
{
    return (x > 0.f) ? x : 0.f;
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

float LPP::activations::Sigmoid::apply_activation_(float x) const
{
    return 1.f / (1.f + std::exp(-x));
}

float LPP::activations::Sigmoid::apply_derivative(float x) const
{
    float s = LPP::activations::Sigmoid::apply_activation_(x);
    return s * (1.f-s);
}

const std::string& LPP::activations::Sigmoid::who() const {
    return STRING_SIGMOID_;
}

float LPP::activations::Tanh::apply_activation_(float x) const
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

void LPP::activations::Softmax::apply_activation(std::vector<float>& v) const
{
    // Provide numerical stability
    float m = *std::max_element(v.begin(), v.end());

    float denom = 0.f;
    for (float& x : v) {
        x = std::exp(x-m);
        denom += x;
    }
    for (float&x : v) {
        x /= denom;
    }
}

void LPP::activations::Softmax::calculate_jacobian(const std::vector<float>& fv, LPP::Matrix& J) const
{
    __lpp_check__(J.rows() == J.cols(),
        "Softmax::calculate_jacobian - given matrix is non-square");
    __lpp_check__(fv.size() == J.rows(),
        "Softmax::calculate_jacobian - vector and matrix sizes disagree");

    size_t n = J.rows();

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            // if (i != j) {
            //     J[i][j] = -1 * fv[i] * fv[j];
            // } else {
            //     J[i][j] = fv[i] * (1.f - fv[i]);
            // }
            J[i][j] = -fv[i]*fv[j];
            if (i == j) J[i][j] += fv[i];
        }
    }
}


float LPP::activations::Softmax::apply_activation_(float x) const
{
    enforce_condition(false,
        "Softmax::apply_activation - this method should not be called element-wise");
    
    return std::nan("");
}

float LPP::activations::Softmax::apply_derivative(float x) const {
    enforce_condition(false,
        "Softmax::apply_derivative - jacobian matrix is non-diagonal");

    return std::nan("");
}

const std::string& LPP::activations::Softmax::who() const {
    return STRING_SOFTMAX_;
}