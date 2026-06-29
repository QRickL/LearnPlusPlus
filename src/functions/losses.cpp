#include "losses.h"
#include "../checking/check.h"
#include <cmath>

LPP::Loss::~Loss() {}

float LPP::MeanSquaredError::apply_loss(const Matrix& y_hat, const Matrix& y) const
{
    __lpp_check__(same_dims(y_hat, y), "MeanSquaredError::apply_loss - y_hat and y different size");

    size_t m = y_hat.rows();
    size_t n = y_hat.cols();
    float sum = 0.0;

    for (size_t i = 0; i < m; i ++) {
        for (size_t j = 0; j < n; j++) {
            sum += pow(y_hat.get(i,j) - y.get(i,j), 2);
        }
    }
    return sum / m;
}

std::vector<float> LPP::MeanSquaredError::find_gradient(const std::vector<float>& y_hat, const std::vector<float>& y) const
{
    __lpp_check__(y_hat.size() == y.size(), "MeanSquaredError::find_gradient - y_hat and y different size");

    return 2 * (y_hat - y);
}

float LPP::BinaryCrossEntropy::apply_loss(const Matrix& y_hat, const Matrix& y) const
{
    __lpp_check__(same_dims(y_hat, y), "BinaryCrossEntropy::apply_loss - y_hat and y different size");
    __lpp_check__(y_hat.cols() == 1, "BinaryCrossEntropy::apply_loss - BCE can only have one response variate");
    
    size_t m = y_hat.rows();
    size_t n = y_hat.cols();
    float sum = 0.0;

    for (size_t i = 0; i < n; i ++) {
        sum += y.get(i, 0) * std::log(y_hat.get(i, 0)) + (1 - y.get(i, 0)) * std::log(1 - y_hat.get(i, 0));
    }
    return -sum / m;
}

std::vector<float> LPP::BinaryCrossEntropy::find_gradient(const std::vector<float>& y_hat, const std::vector<float>& y) const
{
    __lpp_check__(y_hat.size() == y.size(), "BinaryCrossEntropy::find_gradient - y_hat and y different size");
    __lpp_check__(y_hat.size() == 1, "BinaryCrossEntropy::apply_loss - BCE can only have one response variate");

    const float l = (1 - y[0]) / (1 - y_hat[0]) - y[0] / y_hat[0];
    return {l};
}

float LPP::CrossEntropy::apply_loss(const Matrix& y_hat, const Matrix& y) const
{
    __lpp_check__(same_dims(y_hat, y), "CrossEntropy::apply_loss - y_hat and y different size");

    size_t m = y_hat.rows();
    size_t n = y_hat.cols();
    float sum = 0.0;

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            sum += y.get(i,j) * std::log(y_hat.get(i,j));
        }
    }
    return -sum / m;
}

std::vector<float> LPP::CrossEntropy::find_gradient(const std::vector<float>& y_hat, const std::vector<float>& y) const
{
    __lpp_check__(y_hat.size() == y.size(), "CrossEntropy::find_gradient - y_hat and y different size");

    return -1 * (y / y_hat);
}