#ifndef MLP_H
#define MLP_H

#include "network.h"
#include "memory"
#include "exception"
#include "iostream"

typedef double (*activation_ptr)(double);

class MLP_Layer {
    //size_t neuron_count;
    Matrix weights;
    std::vector<double> biases;

    activation_ptr activation_function;

public:
    // Construction using input size and output size and activation function
    MLP_Layer(size_t input_size, size_t output_size, activation_ptr af) : weights{output_size, input_size} {
        biases = std::vector<double>(output_size);
        activation_function = af;
    }

    void display_weight_dimensions() const {
        std::cout << weights.getRows() << 'x' << weights.getCols();
    }

};

class MultiLayerPerceptron : public Network {

    std::vector<std::unique_ptr<MLP_Layer>> layers;
    bool compiled = false;

public:
    // Construct using {neurons in layer, activation function}
    MultiLayerPerceptron(size_t input_size, const std::vector<std::tuple<size_t, activation_ptr>>& given_layers) {
        if (given_layers.empty()) {
            throw std::runtime_error("No layers provided to MLP");
        }

        layers.reserve(given_layers.size());

        size_t cur_input = input_size;
        size_t cur_output;

        for (const auto& l : given_layers) {
            cur_output = std::get<0>(l);

            // Adds layers to the network
            layers.push_back(std::make_unique<MLP_Layer>(cur_input, cur_output, std::get<1>(l)));
            
            cur_input = cur_output;
        }
    }

    void display_mlp_dimensions() const {
        for (const auto& l : layers) {
            l->display_weight_dimensions();
            std::cout << ' ';
        }
        std::cout << std::endl;
    }

    std::vector<double> forward_prop(const std::vector<double>& input) {
        if (!compiled) {
            throw std::runtime_error("Calling an un-compiled model");
        }

        // :D
    }

};

#endif