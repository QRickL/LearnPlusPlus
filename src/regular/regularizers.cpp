#include "regularizer.hpp"
#include "../checking/check.hpp"
#include <cmath>

LPP::regular::Regularizer::~Regularizer() {}

int LPP::sign(float x)
{
    return (x > 0.f) - (x < 0.f);
}

void LPP::regular::LASSO::add_regularization_term_derivative(
    const LPP::Matrix& layer_weights,
    LPP::Matrix& layer_derivatives 
) const {
    size_t m = layer_weights.rows();
    size_t n = layer_weights.cols();

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            layer_derivatives[i][j] += (float)sign(layer_weights[i][j]);
        }
    }
}

void LPP::regular::Ridge::add_regularization_term_derivative(
    const LPP::Matrix& layer_weights,
    LPP::Matrix& layer_derivatives
) const {
    size_t m = layer_weights.rows();
    size_t n = layer_weights.cols();

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            layer_derivatives[i][j] += 2 * layer_weights[i][j];
        }
    }
}

LPP::regular::ElasticNet::ElasticNet(float alpha)
: alpha_{alpha}
{
    enforce_condition(0.f <= alpha_ && alpha_ <= 1.f,
        "ElasticNet::ElasticNet - require 0 <= alpha <= 1");
}

void LPP::regular::ElasticNet::add_regularization_term_derivative(
    const LPP::Matrix& layer_weights,
    LPP::Matrix& layer_derivatives
) const {
    size_t m = layer_weights.rows();
    size_t n = layer_weights.cols();

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float d = layer_weights[i][j];
            layer_derivatives[i][j] += alpha_ * (float)sign(d) + (1 - alpha_) * 2 * d;
        }
    }
}