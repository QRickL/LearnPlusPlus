#include "../tests/lpp_timer.h"
#include "../src/LPP.h"
#include "example_data.h"
#include <cmath>

// The neural network will learn this function on 2000 training examples
float function(float x, float y, float z) {
    return std::pow(x,3) + 3 * y - y * z;
}


int main() {

    // Distribution to generate model weights
    auto my_distribution = std::make_shared<LPP::Normal>(0, 0.7); // mean, stddev

    // Create model
    LPP::Network example_model(
        3,                          // Input size
        {   
            {16, LPP::activations::TANH},
            {16, LPP::activations::TANH},
            {1,  LPP::activations::IDENTITY}
        },
        my_distribution             // Weights sampled when network is created
    );

    // Train the model!
    LPP::start_timer();
    example_model.train(explanatory_variates, response_variates, 1000, 0.01, LPP::losses::MEAN_SQUARED_ERROR);  // Dont worry about overfitting because of simple dataset
    LPP::end_timer();

    // See how it did on training data
    std::cout << "First 10 examples:" << std::endl;
    for (int i = 0; i < 10; i++) {
        auto res      = example_model.inference(explanatory_variates[i]);
        float actual = function(explanatory_variates[i][0], explanatory_variates[i][1], explanatory_variates[i][2]);

        std::cout << "Actual:  " << actual << std::endl;
        std::cout << "Predict: " << res[0] << std::endl << std::endl;
    }

    std::vector<float> v1 = {0, 0, 0};
    std::vector<float> v2 = {-2.1, 0.512, 1.2};
    std::vector<float> v3 = {0.2132, 3.123, -1.112};

    // See how it does on new data
    std::cout << "New data:" << std::endl;

    std::cout << "Actual:  " << function(v1[0], v1[1], v1[2]) << std::endl;
    auto u1 = example_model.inference(v1);
    std::cout << "Predict: " << u1[0] << std::endl << std::endl;

    std::cout << "Actual:  " << function(v2[0], v2[1], v2[2]) << std::endl;
    auto u2 = example_model.inference(v2);
    std::cout << "Predict: " << u2[0] << std::endl << std::endl;

    std::cout << "Actual:  " << function(v3[0], v3[1], v3[2]) << std::endl;
    auto u3 = example_model.inference(v3);
    std::cout << "Predict: " << u3[0] << std::endl << std::endl;
}