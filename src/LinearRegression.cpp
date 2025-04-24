#include "LinearRegression.h"

LinearRegression::LinearRegression() : numFeatures{0}, bias{0}, coefficients{} {}

inline double LinearRegression::singleton(const std::vector<double> &features) {
    return bias + dot(coefficients, features);
}

void LinearRegression::fit(const std::vector<std::vector<double>> &trainingData, const std::vector<double> &labels, const std::size_t epochs) {
    // TO DO: exceptions

    // Allow for multiple trainings
    // Parameters should not be reset in subsequent trainings
    if (numFeatures == 0) {
        numFeatures = trainingData[0].size();
        coefficients = std::vector<double>(numFeatures, 0);
    }

    for (std::size_t e = 0; e < epochs; e++) {
        // TO DO: update bias and coefficients
        // Figure out how I want to this cleanly
    }
}

std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>> &inputs) {
    if (inputs.size() == 0) {
        throw std::invalid_argument("No inputs given to predict on");
    }

    std::vector<double> predictions(inputs.size());

    for (int i = 0; i < inputs.size(); i++) {
        // If an input has the wrong number of features, dot() will detect it
        predictions[i] = singleton(inputs[i]);
    }
    return predictions;
}

double LinearRegression::getBias() {
    if (coefficients.size() == 0) {
        throw std::logic_error("Model has not yet been trained");
    }

    return bias;
}

std::vector<double> LinearRegression::getCoefficients() {
    if (coefficients.size() == 0) {
        throw std::logic_error("Model has not yet been trained");
    }
    
    return coefficients;
}
