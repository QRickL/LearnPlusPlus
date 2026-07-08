#ifndef LPP_LAYER_HPP
#define LPP_LAYER_HPP

#include "functions/activations.hpp"
#include "mathobj/matrix.hpp"
#include <vector>
#include <memory>

namespace LPP {

class Layer {
    using Vec = std::vector<float>;

    Matrix  weights_;
    Vec     biases_;
    mutable Vec pre_activation_vals_;     // Needed for backpropagation, holds z = Wx + b
    mutable Vec post_activation_vals_;    // Needed for backpropagation, holds a = σ(z)
                                          // Two vectors are allocated at construction. Copying does not use extra allocation
    const activations::Activation* activation_func_; // Pointer to activation function used by layer

    friend class Network;

public:
    // Constructor called when network created manually
    // If no probability distribution is provided, then the network initializes with He initialization
    Layer(
        size_t input_size,
        size_t output_size,
        const activations::Activation* af,
        distribution::ProbabilityDistribution* pd = nullptr
    );

    // Constructor called when network created from file
    Layer(
        Matrix& given_weights,
        Vec& given_biases,
        const activations::Activation* af
    );

    // Display information
    void display(std::ostream& os = std::cout) const;
};

// Display information
void print_object(const Layer& l, std::ostream& os = std::cout);

} // namespace LPP
#endif
