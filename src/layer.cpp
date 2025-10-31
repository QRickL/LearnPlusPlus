#include "layer.h"
#include <iostream>

LPP::Layer::Layer(const size_t input_size, const size_t output_size, const std::shared_ptr<Activation>& af, const std::shared_ptr<ProbabilityDistribution>& pd)
{
    if (pd) {
        weights = std::make_unique<Matrix>(output_size, input_size, pd);
        biases = std::make_unique<std::vector<double>>(output_size);
        for (size_t i = 0; i < output_size; i++) {
            (*biases)[i] = pd->sample();
        }
    } else {
        weights = std::make_unique<Matrix>(output_size, input_size);
        biases = std::make_unique<std::vector<double>>(output_size, 0.0);
    }
    act_func = af;
}

LPP::Layer::Layer(std::unique_ptr<Matrix>& given_weights, std::unique_ptr<std::vector<double>>& given_biases, std::shared_ptr<Activation>& af)
{
    weights = std::move(given_weights);
    biases = std::move(given_biases);
    act_func = af;
}

// TODO: parallelize
void LPP::Layer::apply_activation(std::vector<double>& z) const
{
    for (size_t i = 0; i < z.size(); i++) {
       z[i] = act_func->apply_itself(z[i]);
    }
}

void LPP::Layer::display() const {
    std::cout << "Weights:\n";
    print_object(*weights);

    std::cout << "Biases:\n";
    print_object(*biases);

    std::cout << "Activation: " << act_func->who() << std::endl;
}

void LPP::print_object(const Layer& l)
{
    l.display();
}
