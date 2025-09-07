#include "mlp.h"

// ***
// Multilayer perceptron layer
// ***

MLP_Layer::MLP_Layer(size_t input_size, size_t output_size, activation_ptr af)
    : weights{output_size, input_size}
{
    biases = std::vector<double>(output_size);
    activation_function = af;
}

void MLP_Layer::display_weight_dimensions() const {
    std::cout << weights.getRows() << 'x' << weights.getCols();
}

void MLP_Layer::display_weights() const {
    std::cout << "Weights:\n"; 
    weights.display_entries();
    std::cout << '\n';

    std::cout << "Biases:\n";
    for (const auto d : biases) {
        std::cout << d << '\n';
    }
}

// ***
// Multilayer perceptron
// ***

MultiLayerPerceptron::MultiLayerPerceptron(size_t input_size, const std::vector<std::tuple<size_t, activation_ptr>>& given_layers) {
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

void MultiLayerPerceptron::display_mlp_dimensions() const {
    for (const auto& l : layers) {
        l->display_weight_dimensions();
        std::cout << ' ';
    }
    std::cout << std::endl;
}

void MultiLayerPerceptron::display_mlp() const {
    int i = 1;
    for (const auto& l : layers) {
        std::cout << "Layer " << i << ":\n";
        l->display_weights();
        std::cout << '\n';

        i++;
    }
}

// currently working on
std::vector<double> MultiLayerPerceptron::forward_prop_train(std::vector<double> activation) {
    /*
    // set up safety when everything else is done
    if (!compiled) {
        throw std::runtime_error("Calling an un-compiled model");
    }
    */

    for (const auto& l : layers) {
        activation = add_vectors(l->weights * activation, l->biases);

        if (l->activation_function) {
            // ***
            // TO DO: parallelize this
            // ***
            for (auto& a : activation) {
                a = l->activation_function(a);
            }
        }

        // store intermediate values for backpropagation
        // apply the activation function to activation in parallel fashion
    }

    return activation;
}