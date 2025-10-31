#ifndef LPP_LAYER_H
#define LPP_LAYER_H

#include "functions/activations.h"
#include "matrix.h"
#include <vector>
#include <memory>

namespace LPP {

class Layer {
    std::unique_ptr<Matrix> weights;
    std::unique_ptr<std::vector<double>> biases;
    std::shared_ptr<Activation> act_func;
    std::vector<double> pre_activation;     // Needed for backpropagation, holds z = Wx + b
    std::vector<double> post_activation;    // Needed for backpropagation, holds a = σ(z)

    // Apply activation function to all entries, performed in place in z
    void apply_activation(std::vector<double>& z) const;

    friend class Network;

public:
    // Constructor called when network created manually
    Layer(
        const size_t input_size,
        const size_t output_size,
        const std::shared_ptr<Activation>& af,
        const std::shared_ptr<ProbabilityDistribution>& pd
    );

    // Constructor called when network created from file
    Layer(
        std::unique_ptr<Matrix>& given_weights,
        std::unique_ptr<std::vector<double>>& given_biases,
        std::shared_ptr<Activation>& af
    );

    // Display information
    void display() const;
};

// Display information
void print_object(const Layer& l);

} // namespace LPP
#endif
