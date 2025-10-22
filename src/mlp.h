#ifndef LPP_MLP_H
#define LPP_MLP_H

#include <memory>
#include <iostream>
#include "layer.h"
#include "network.h"
#include "matrix.h"

namespace LPP {

class MLP_Layer : public Layer {
    std::unique_ptr<Matrix> weights;
    std::unique_ptr<std::vector<double>> biases;
    std::vector<double> intermed_val;  // Used for backpropagation
    std::shared_ptr<Activation> act_func;

    friend class MLP;

public:
    // Construct with in size, out size, and pointer to activation function
    MLP_Layer(const size_t input_size, const size_t output_size, std::shared_ptr<Activation> af = IDENTITY);

    // Construct when reading from model file
    MLP_Layer(std::unique_ptr<Matrix>& given_weights, std::unique_ptr<std::vector<double>>& given_biases, std::shared_ptr<Activation> af = IDENTITY);

    // Display information
    void display() const;

    // Apply activation function to all entries
    void apply_activation(std::vector<double>& x) const;

    ~MLP_Layer() {}
};

class MLP : public Network {
    std::vector<std::unique_ptr<MLP_Layer>> layers;

    // Used to perform backpropagation
    //std::vector<std::vector<double>> intermed_values;
    // Is this truly const, if I'm changing the values of the things pointing to?
    std::vector<double> forward_propagation(const std::vector<double>&x, const bool saving) const;

    // calculate gradients

    // backpropagate

    // training = calculate it for everything then update weights. repeat

public:
    // Manually specify model architecture
    MLP(
        const size_t input_size,
        const std::vector<std::pair<size_t, std::shared_ptr<Activation>>> s
    );

    // Load model + weights from file
    MLP(const std::string& filepath);

    // Fire through network, will not save intermediate values
    // Not used for traninig
    std::vector<double> inference(const std::vector<double>& layer_info) const;

    // Saving weights to a file
    void save_model(const std::string& filepath) const override;

    ~MLP() {}
};

} // namespace LPP

#endif