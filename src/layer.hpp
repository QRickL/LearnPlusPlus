#ifndef LPP_LAYER_HPP
#define LPP_LAYER_HPP

#include "functions/activations.hpp"
#include "mathobj/matrix.hpp"
#include <vector>
#include <memory>

namespace LPP {

class Layer {

    using Vec = std::vector<float>;
    template <typename T>
    using up = std::unique_ptr<T>;

    up<Matrix> weights_;
    up<Vec>    biases_;
    Vec        pre_activation_vals_;     // Needed for backpropagation, holds z = Wx + b
    Vec        post_activation_vals_;    // Needed for backpropagation, holds a = σ(z)
    const activations::Activation* activation_func_; // Pointer to activation function used by layer

    void apply_activation_layer_(Vec& z) const;    // Apply activation function to all entries, performed in place in z

    friend class Network;

public:
    // Constructor called when network created manually
    Layer(
        size_t input_size,
        size_t output_size,
        const activations::Activation* af,
        distribution::ProbabilityDistribution* pd
    );

    // Constructor called when network created from file
    Layer(
        up<Matrix>& given_weights,
        up<Vec>& given_biases,
        const activations::Activation* af
    );

    // Display information
    void display(std::ostream& os = std::cout) const;
};

// Display information
void print_object(const Layer& l, std::ostream& os = std::cout);

} // namespace LPP
#endif
