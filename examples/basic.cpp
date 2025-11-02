#include "../tests/lpp_timer.h"
#include "../src/LPP.h"
#include "example_data.h"
#include <cmath>

// The neural network will learn this function on 2000 training examples
double function(double x, double y, double z) {
    return std::pow(x,3) + 3 * y - y * z;
}

int main() {

    // Distribution to generate model weights
    const auto my_distribution = std::make_shared<LPP::Normal>(0, 0.5); // mean, stddev

    // Create model
    LPP::Network example_model(
        3,
        {   
            {20, LPP::RELU},
            {15, LPP::RELU},
            {10,  LPP::RELU},
            {1,  LPP::IDENTITY}
        },
        my_distribution
    );

    // Train the model!
    LPP::start_timer();
    example_model.train(explanatory_variates, response_variates, 100, 0.005, LPP::MEAN_SQUARED_ERROR);
    LPP::end_timer();

    // See how it did on training data
    std::cout << "First 10 examples:" << std::endl;
    for (int i = 0; i < 10; i++) {
        const auto res      = example_model.inference(explanatory_variates[i]);
        const double actual = function(explanatory_variates[i][0], explanatory_variates[i][1], explanatory_variates[i][2]);

        std::cout << "Actual:  " << actual << std::endl;
        std::cout << "Predict: " << res[0] << std::endl << std::endl;
    }

    const std::vector<double> v1 = {0, 0, 0};
    const std::vector<double> v2 = {-2.1, 0.512, 1.2};
    const std::vector<double> v3 = {0.2132, 3.123, -1.112};

    // See how it does on new data
    std::cout << "New data:" << std::endl;

    std::cout << "Actual:  " << function(v1[0], v1[1], v1[2]) << std::endl;
    const auto u1 = example_model.inference(v1);
    std::cout << "Predict: " << u1[0] << std::endl << std::endl;

    std::cout << "Actual:  " << function(v2[0], v2[1], v2[2]) << std::endl;
    const auto u2 = example_model.inference(v2);
    std::cout << "Predict: " << u2[0] << std::endl << std::endl;

    std::cout << "Actual:  " << function(v3[0], v3[1], v3[2]) << std::endl;
    const auto u3 = example_model.inference(v3);
    std::cout << "Predict: " << u3[0] << std::endl << std::endl;
}