#include "layer.hpp"
#include "checking/check.hpp"
#include <iostream>

LPP::Layer::Layer(
    size_t input_size,
    size_t output_size,
    const activations::Activation* af,
    distribution::ProbabilityDistribution* pd
) :
    activation_func_{af},
    // Use std::copy when copying to avoid reallocations during training
    pre_activation_vals_{std::vector<float>(output_size)},
    post_activation_vals_{std::vector<float>(output_size)}
{
    enforce_condition(af, "Layer::Layer - provided activation function is nullptr");

    if (pd != nullptr) {
        weights_ = std::make_unique<Matrix>(output_size, input_size, pd);
    } else {
        weights_ = std::make_unique<Matrix>(output_size, input_size);
    }
    biases_ = std::make_unique<std::vector<float>>(output_size, 0.f); // biases typically initialized as 0
}

LPP::Layer::Layer(
    std::unique_ptr<Matrix>& given_weights,
    std::unique_ptr<std::vector<float>>& given_biases,
    const activations::Activation* af
) :
    weights_{std::move(given_weights)},
    biases_{std::move(given_biases)},
    activation_func_{af},
    // Use std::copy when copying to avoid reallocations during training
    pre_activation_vals_{std::vector<float>(weights_->rows())},
    post_activation_vals_{std::vector<float>(weights_->rows())}
{}

void LPP::Layer::display(std::ostream& os) const {
    os << "Weights:\n";
    print_object(*weights_, os);

    os << "Biases:\n";
    print_object(*biases_, os);

    os << "Activation: " << activation_func_->who() << std::endl;
}

void LPP::print_object(const Layer& l, std::ostream& os) {l.display(os);}
