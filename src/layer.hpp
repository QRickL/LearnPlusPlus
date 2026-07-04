#ifndef LPP_LAYER_H
#define LPP_LAYER_H

#include "functions/activations.hpp"
#include "mathobj/matrix.hpp"
#include <vector>
#include <memory>

namespace LPP {

class Layer {
    std::unique_ptr<Matrix>             weights_;
    std::unique_ptr<std::vector<float>> biases_;
    std::vector<float>                  pre_activation_vals_;     // Needed for backpropagation, holds z = Wx + b
    std::vector<float>                  post_activation_vals_;    // Needed for backpropagation, holds a = σ(z)
    std::shared_ptr<Activation>         activation_func_;         // Pointer to activation function used by layer
    void apply_activation_layer_(std::vector<float>& z) const;    // Apply activation function to all entries, performed in place in z

    friend class Network;

public:
    // Constructor called when network created manually
    Layer(
        size_t input_size,
        size_t output_size,
        const std::shared_ptr<Activation>& af,
        const std::shared_ptr<ProbabilityDistribution>& pd
    );

    // Constructor called when network created from file
    Layer(
        std::unique_ptr<Matrix>& given_weights,
        std::unique_ptr<std::vector<float>>& given_biases,
        std::shared_ptr<Activation>& af
    );

    // Display information
    void display() const;
};

// Display information
void print_object(const Layer& l);

} // namespace LPP
#endif
