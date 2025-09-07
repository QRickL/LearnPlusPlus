#ifndef MLP_H
#define MLP_H

#include "network.h"
#include "layer.h"
#include "memory"
#include "exception"
#include "iostream"

typedef double (*activation_ptr)(double);

class MLP_Layer : public Layer{
    Matrix weights;
    std::vector<double> biases;
    activation_ptr activation_function;
    friend class MultiLayerPerceptron;

public:
    // Construct using input size and output size and activation function
    MLP_Layer(size_t input_size, size_t output_size, activation_ptr af);
    void display_weight_dimensions() const;
    void display_weights() const;
};

class MultiLayerPerceptron : public Network {
    std::vector<std::unique_ptr<MLP_Layer>> layers;
    bool compiled = false;

public:
    // Construct using {#neurons in layer, ptr to activation function}
    MultiLayerPerceptron(size_t input_size, const std::vector<std::tuple<size_t, activation_ptr>>& given_layers);
    void display_mlp_dimensions() const;
    void display_mlp() const;

    // currently working on
    std::vector<double> forward_prop_train(std::vector<double> activation);
    // have a separate one for normal inference, with no need for storing intermediate values

};

#endif