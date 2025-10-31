#include "losses.h"
#include <cmath>

LPP::Loss::~Loss() {}

void LPP::Loss::enforce_size(const Matrix& y_hat, const Matrix& y, const std::string& loss_name) const
{
    if (!same_dims(y_hat, y)) {
        const auto msg = "Matrix dimensions for application of " + loss_name + " do not match";
        throw std::invalid_argument(msg);
    }
}

void LPP::Loss::enforce_size(const std::vector<double>& y, const std::vector<double>& y_hat, const std::string& loss_name) const
{
    if (y_hat.size() != y.size()) {
        const auto msg = "Vector dimensions for application of " + loss_name + " do not match";
        throw std::invalid_argument(msg);
    }
}

// TODO: parallelize this
double LPP::MeanSquaredError::apply_itself(const Matrix& y_hat, const Matrix& y) const
{
    enforce_size(y_hat, y, "MeanSquaredError");

    const size_t m = y_hat.rows();
    const size_t n = y_hat.cols();
    double sum = 0.0;

    for (size_t i = 0; i < m; i ++) {
        for (size_t j = 0; j < n; j++) {
            sum += pow(y_hat.get(i,j) - y.get(i,j), 2);
        }
    }
    return sum / m;
}

std::vector<double> LPP::MeanSquaredError::find_gradient(const std::vector<double>& y_hat, const std::vector<double>& y) const
{
    enforce_size(y_hat, y, "MeanSquaredError derivative");

    return 2 * (y_hat - y);
}

double LPP::BinaryCrossEntropy::apply_itself(const Matrix& y_hat, const Matrix& y) const
{
    enforce_size(y_hat, y, "BinaryCrossEntropy");
    if (y_hat.cols() != 1) {
        auto msg = "Application of BinaryCrossEntropy must take in one response variate (yes/no)";
        throw std::invalid_argument(msg);
    }

    const size_t m = y_hat.rows();
    const size_t n = y_hat.cols();
    double sum = 0.0;

    for (size_t i = 0; i < n; i ++) {
        sum += y.get(i, 0) * std::log(y_hat.get(i, 0)) + (1 - y.get(i, 0)) * std::log(1 - y_hat.get(i, 0));
    }
    return -sum / m;
}

std::vector<double> LPP::BinaryCrossEntropy::find_gradient(const std::vector<double>& y_hat, const std::vector<double>& y) const
{
    enforce_size(y_hat, y, "BinaryCrossEntropy derivative");
    if (y_hat.size() != 1) {
        auto msg = "Application of BinaryCrossEntropy derivative must take in one response variate (yes/no)";
        throw std::invalid_argument(msg);
    }

    const double l = (1 - y[0]) / (1 - y_hat[0]) - y[0] / y_hat[0];
    return {l};
}

double LPP::CrossEntropy::apply_itself(const Matrix& y_hat, const Matrix& y) const
{
    enforce_size(y_hat, y, "CrossEntropy");

    const size_t m = y_hat.rows();
    const size_t n = y_hat.cols();
    double sum = 0.0;

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            sum += y.get(i,j) * std::log(y_hat.get(i,j));
        }
    }
    return -sum / m;
}

std::vector<double> LPP::CrossEntropy::find_gradient(const std::vector<double>& y_hat, const std::vector<double>& y) const
{
    enforce_size(y_hat, y, "CrossEntropy derivative");

    return -1 * (y / y_hat);
}