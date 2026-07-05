#ifndef LPP_REGULARIZER_HPP
#define LPP_REGULARIZER_HPP

#include "../mathobj/matrix.hpp"

namespace LPP {
namespace regular {

class Regularizer {
public:
    virtual void add_regularization_term_derivative(
        const Matrix& layer_weights,
        Matrix& layer_derivatives
    ) const = 0;

    virtual float add_regularization_loss_penalty(
        const Matrix& layer_weights
    ) const = 0;

    virtual ~Regularizer() = 0;
};


class LASSO : public Regularizer {
public:
    void add_regularization_term_derivative(
        const Matrix& layer_weights,
        Matrix& layer_derivatives
    ) const override;

    float add_regularization_loss_penalty(
        const Matrix& layer_weights
    ) const override;

    LASSO();
    ~LASSO() {}
};

class Ridge : public Regularizer {
public:
    void add_regularization_term_derivative(
        const Matrix& layer_weights,
        Matrix& layer_derivatives
    ) const override;

    float add_regularization_loss_penalty(
        const Matrix& layer_weights
    ) const override;

    Ridge();
    ~Ridge() {}
};

class ElasticNet : public Regularizer {
    const float alpha_;
public:
    void add_regularization_term_derivative(
        const Matrix& layer_weights,
        Matrix& layer_derivatives
    ) const override;

    float add_regularization_loss_penalty(
        const Matrix& layer_weights
    ) const override;

    ElasticNet(float alpha);
    ~ElasticNet() {}
};

}

int sign(float x);

}

#endif