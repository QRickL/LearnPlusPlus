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
            float d = y_hat[i][j] - y[i][j];
            sum += d*d;
        }
    }
    return sum / m;
}

std::vector<float> LPP::loss::MeanSquaredError::find_gradient(const std::vector<float>& y_hat, const std::vector<float>& y) const
{
    __lpp_check__(y_hat.size() == y.size(),
        "MeanSquaredError::find_gradient - y_hat and y different size");
    enforce_condition(false,
        "MeanSqauredError::find_gradient - this function is deprecated");

    return 2.f * (y_hat - y);
}

void LPP::loss::MeanSquaredError::find_gradient_then_write(const std::vector<float>& y_hat, const std::vector<float>& y, std::vector<float>& to_write) const
{
    __lpp_check__(y_hat.size() == y.size(),
        "MeanSquaredError::find_gradient_then_write - y_hat and y different size");
    __lpp_check__(to_write.size() == y.size(),
        "MeanSquaredError::find_gradient_then_write - to_write and y different size");

    for (size_t i = 0; i < y.size(); i++)
    {
        to_write[i] = 2.f * (y_hat[i] - y[i]);
    }
}

float LPP::loss::BinaryCrossEntropy::apply_loss(const Matrix& y_hat, const Matrix& y) const
{
    __lpp_check__(same_dims(y_hat, y),
        "BinaryCrossEntropy::apply_loss - y_hat and y different size");
    __lpp_check__(y_hat.cols() == 1,
        "BinaryCrossEntropy::apply_loss - BCE can only have one response variate");
    
    size_t m = y_hat.rows();
    size_t n = y_hat.cols();
    float sum = 0.f;

    for (size_t i = 0; i < m; i ++) {
        float y_val = clamp(y[i][0], loss_epsilon, 1.f - loss_epsilon);
        float y_hat_val = clamp(y_hat[i][0], loss_epsilon, 1.f - loss_epsilon);

        sum += y_val * std::log(y_hat_val) + (1 - y_val) * std::log(1 - y_hat_val);
    }
    return -sum / m;
}

std::vector<float> LPP::loss::BinaryCrossEntropy::find_gradient(const std::vector<float>& y_hat, const std::vector<float>& y) const
{
    __lpp_check__(y_hat.size() == y.size(),
        "BinaryCrossEntropy::find_gradient - y_hat and y different size");
    __lpp_check__(y_hat.size() == 1,
        "BinaryCrossEntropy::find_gradient - BCE can only have one response variate");
    enforce_condition(false,
        "BinaryCrossEntropy::find_gradient - this function is deprecated");

    float y_val = clamp(y[0], loss_epsilon, 1.f - loss_epsilon);
    float y_hat_val = clamp(y_hat[0], loss_epsilon, 1.f - loss_epsilon);

    const float l = (1.f - y_val) / (1.f - y_hat_val) - y_val / y_hat_val;
    return {l};
}

void LPP::loss::BinaryCrossEntropy::find_gradient_then_write(const std::vector<float>& y_hat, const std::vector<float>& y, std::vector<float>& to_write) const
{
    __lpp_check__(y_hat.size() == y.size(),
        "BinaryCrossEntropy::find_gradient_then_write - y_hat and y different size");
    __lpp_check__(y_hat.size() == 1,
        "BinaryCrossEntropy::find_gradient_then_write - BCE can only have one response variate");
    __lpp_check__(to_write.size() == 1,
        "BinaryCrossEntropy::find_gradient_then_write - to_write can only have size 1");

    float y_val = clamp(y[0], loss_epsilon, 1.f - loss_epsilon);
    float y_hat_val = clamp(y_hat[0], loss_epsilon, 1.f - loss_epsilon);
    to_write[0] = (1.f - y_val) / (1.f - y_hat_val) - y_val / y_hat_val;
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
            sum += (y[i][j] != 0.f) ? y[i][j] * std::log(clamp(y_hat[i][j], loss_epsilon, 1.f - loss_epsilon)) : 0.f;
        }
    }
    return -sum / m;
}

std::vector<float> LPP::loss::CrossEntropy::find_gradient(const std::vector<float>& y_hat, const std::vector<float>& y) const
{
    __lpp_check__(y_hat.size() == y.size(),
        "CrossEntropy::find_gradient - y_hat and y different size");
    enforce_condition(false,
        "CrossEntropy::find_gradient - this function is deprecated");

    return -1 * (y / y_hat);
}

void LPP::loss::CrossEntropy::find_gradient_then_write(const std::vector<float>& y_hat, const std::vector<float>& y, std::vector<float>& to_write) const
{
    __lpp_check__(y_hat.size() == y.size(),
        "CrossEntropy::find_gradient_then_write - y_hat and y different size");
    __lpp_check__(to_write.size() == y.size(),
        "CrossEntropy::find_gradient_then_write - to_write and y different size");

    for (size_t i = 0; i < y.size(); i++)
    {
        to_write[i] = - y[i] / y_hat[i];
    }
}

float LPP::clamp(float x, float lower, float upper)
{
    return std::max(lower, std::min(x, upper));
}