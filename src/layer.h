#pragma once

#include "functions/activations.h"
#include "matrix.h"
#include <iostream>
#include <vector>
#include <memory>

namespace LPP {

template <typename number>
class Layer {
    using Weights       = std::unique_ptr<Matrix<number>>;
    using Biases        = std::unique_ptr<Vect<number>>;
    using ActivationPtr = std::shared_ptr<Activation<number>>;
    using ProbDistnPtr  = std::shared_ptr<ProbabilityDistribution>;

    Weights         weights__;
    Biases          biases__;
    Vect<number>    pre_activation__;     // Needed for backpropagation, holds z = Wx + b
    Vect<number>    post_activation__;    // Needed for backpropagation, holds a = σ(z)
    ActivationPtr   activation_func__;    // Will be applied after matrix vector multiplication

    void apply_activation_layer__(Vect<number>& z) const; // Apply activation function to all entries, performed in place in z
    friend class Network;

public:
    // Layer constructor called when network created manually
    Layer(
        size_t inputSize,
        size_t outputSize,
        ActivationPtr& initActivationFunction, // TODO: can pass in a string instead
        const ProbDistnPtr&  initProbDistn
    );

    // Layer constructor called when network created from file
    Layer(
        Weights&        initWeights,
        Biases&         initBiases,
        ActivationPtr&  initActivationFunction
    );

    // Prints contents of the layer: weights, biases, and activation
    void printLayer(std::ostream& os = std::cout) const;
};

} // namespace LPP
