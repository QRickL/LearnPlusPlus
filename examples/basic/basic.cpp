#include "../../src/LPP.hpp"
#include "data.hpp"
#include <cmath>

// The neural network will learn this function on 2000 training examples
float function(float x, float y, float z) {
    return x*x*x + 3*y - y*z;
}


int main() {

    // Distribution to initialize model weights
    auto my_distribution = LPP::distribution::Normal(0, 0.2);

    // Create model
    LPP::Network example_model(
        3,  // Network input size
        {
            {16, &LPP::activations::tanh},  // {layer output size, activation function}
            {16, &LPP::activations::tanh},
            {1,  &LPP::activations::identity}
        },
        &my_distribution
    );

    // Train the model!
    example_model.train(
        explanatory_variates,           // Features 
        response_variates,              // Responses
        500,                            // Number of epochs
        0.005,                          // Learning rate
        &LPP::loss::mean_squared_error  // Loss function
    );

    // See how it did on training data
    std::cout << "First 10 examples:" << std::endl;
    for (int i = 0; i < 10; i++) {
        auto res      = example_model.inference(explanatory_variates[i]);
        float actual = function(explanatory_variates[i][0], explanatory_variates[i][1], explanatory_variates[i][2]);

        std::cout << "Actual:  " << actual << std::endl;
        std::cout << "Predict: " << res[0] << std::endl << std::endl;
    }

    // See how it does on new data
    std::cout << "New data:" << std::endl;
    std::vector<std::vector<float>> new_data = {
        {0, 0, 0},
        {-2.1, 0.512, 1.2},
        {0.2132, 3.123, -1.112}
    };
    for (auto v : new_data) {
        std::cout << "Actual: " << function(v[0], v[1], v[2]) << '\n';
        auto estimate = example_model.inference(v);
        std::cout << "Predict: " << estimate[0] << std::endl << std::endl;
    }
}
