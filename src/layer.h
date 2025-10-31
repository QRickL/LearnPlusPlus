#ifndef LPP_LAYER_H
#define LPP_LAYER_H

#include "functions/activations.h"
#include "layer.h"
#include "matrix.h"
#include <vector>
#include <memory>

namespace LPP {

class Layer {
    std::unique_ptr<Matrix> weights;
    std::unique_ptr<std::vector<double>> biases;
    std::shared_ptr<Activation> act_func;

    std::vector<double> pre_activation;     // Needed for backpropagation
    std::vector<double> post_activation;    // Needed for backpropagation

    friend class Network;

public:
    // Construct with in size, out size, and pointer to activation function
    Layer(const size_t input_size, const size_t output_size, std::shared_ptr<Activation> af = IDENTITY);

    // Construct when reading from model file
    Layer(std::unique_ptr<Matrix>& given_weights, std::unique_ptr<std::vector<double>>& given_biases, std::shared_ptr<Activation> af = IDENTITY);

    // Display information
    void display() const;

    // Apply activation function to all entries
    std::vector<double> apply_activation(const std::vector<double>& z) const;
};

} // namespace LPP
#endif
