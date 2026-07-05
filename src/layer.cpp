#include "layer.hpp"
#include <iostream>

LPP::Layer::Layer(
    size_t input_size,
    size_t output_size,
    const std::shared_ptr<activations::Activation>& af,
    const std::shared_ptr<distribution::ProbabilityDistribution>& pd
) {
    if (pd != nullptr) {
        weights_ = std::make_unique<Matrix>(output_size, input_size, pd);

        biases_ = std::make_unique<std::vector<float>>(output_size);
        for (size_t i = 0; i < output_size; i++) {
            (*biases_)[i] = pd->sample();
        }
    } else {
        weights_ = std::make_unique<Matrix>(output_size, input_size);
        biases_ = std::make_unique<std::vector<float>>(output_size, 0.0);
    }
    activation_func_ = af;
}

LPP::Layer::Layer(std::unique_ptr<Matrix>& given_weights, std::unique_ptr<std::vector<float>>& given_biases, std::shared_ptr<activations::Activation>& af)
{
    weights_ = std::move(given_weights);
    biases_ = std::move(given_biases);
    activation_func_ = af;
}

// TODO: this can be parallelized
void LPP::Layer::apply_activation_layer_(std::vector<float>& z) const
{
    for (size_t i = 0; i < z.size(); i++) {
       z[i] = activation_func_->apply_activation(z[i]);
    }
}

void LPP::Layer::display(std::ostream& os) const {
    os << "Weights:\n";
    print_object(*weights_, os);

    os << "Biases:\n";
    print_object(*biases_, os);

    os << "Activation: " << activation_func_->who() << std::endl;
}

void LPP::print_object(const Layer& l, std::ostream& os) {l.display(os);}
