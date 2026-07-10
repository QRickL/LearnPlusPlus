#include "losses.hpp"
#include "../checking/check.hpp"
#include <cmath>

LPP::loss::Loss::~Loss() {}

float LPP::loss::MeanSquaredError::apply_loss(const Matrix& y_hat, const Matrix& y) const
{
    __lpp_check__(same_dims(y_hat, y),
        "MeanSquaredError::apply_loss - y_hat and y different size");

    size_t m = y_hat.rows();
    size_t n = y_hat.cols();
    float sum = 0.0;

    for (size_t i = 0; i < m; i ++) {
        for (size_t j = 0; j < n; j++) {
            float d = y_hat.get(i,j) - y.get(i,j);
            sum += d*d;
        }
    }
    return sum / m;
}

std::vector<float> LPP::loss::MeanSquaredError::find_gradient(const std::vector<float>& y_hat, const std::vector<float>& y) const
{
    __lpp_check__(y_hat.size() == y.size(),
        "MeanSquaredError::find_gradient - y_hat and y different size");

    return 2.f * (y_hat - y);
    //return (2.f / (float)y_hat.size()) * (y_hat - y);
}

float LPP::loss::BinaryCrossEntropy::apply_loss(const Matrix& y_hat, const Matrix& y) const
{
    __lpp_check__(same_dims(y_hat, y),
        "BinaryCrossEntropy::apply_loss - y_hat and y different size");
    __lpp_check__(y_hat.cols() == 1,
        "BinaryCrossEntropy::apply_loss - BCE can only have one response variate");
    
    size_t m = y_hat.rows();
    size_t n = y_hat.cols();
    float sum = 0.0;

    for (size_t i = 0; i < m; i ++) {
        float y_val = clamp(y.get(i, 0), loss_epsilon, 1.f - loss_epsilon);
        float y_hat_val = clamp(y_hat.get(i,0), loss_epsilon, 1.f - loss_epsilon);

        sum += y_val * std::log(y_hat_val) + (1 - y_val) * std::log(1 - y_hat_val);
    }
    return -sum / m;
}

std::vector<float> LPP::loss::BinaryCrossEntropy::find_gradient(const std::vector<float>& y_hat, const std::vector<float>& y) const
{
    __lpp_check__(y_hat.size() == y.size(),
        "BinaryCrossEntropy::find_gradient - y_hat and y different size");
    __lpp_check__(y_hat.size() == 1,
        "BinaryCrossEntropy::apply_loss - BCE can only have one response variate");

    float y_val = clamp(y[0], loss_epsilon, 1.f - loss_epsilon);
    float y_hat_val = clamp(y_hat[0], loss_epsilon, 1.f - loss_epsilon);

    const float l = (1.f - y_val) / (1.f - y_hat_val) - y_val / y_hat_val;
    return {l};
}

float LPP::loss::CrossEntropy::apply_loss(const Matrix& y_hat, const Matrix& y) const
{
    __lpp_check__(same_dims(y_hat, y),
        "CrossEntropy::apply_loss - y_hat and y different size");

    size_t m = y_hat.rows();
    size_t n = y_hat.cols();
    float sum = 0.0;

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            sum += y.get(i,j) * std::log(clamp(y_hat.get(i,j), loss_epsilon, 1.f - loss_epsilon));
        }
    }
    return -sum / m;
}

std::vector<float> LPP::loss::CrossEntropy::find_gradient(const std::vector<float>& y_hat, const std::vector<float>& y) const
{
    __lpp_check__(y_hat.size() == y.size(),
        "CrossEntropy::find_gradient - y_hat and y different size");

    return -1 * (y / y_hat);
}

float LPP::clamp(float x, float lower, float upper)
{
    return std::max(lower, std::min(x, upper));
}