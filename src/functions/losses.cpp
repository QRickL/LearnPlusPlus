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


double LPP::MeanSquaredError::apply_itself(const Matrix& y_hat, const Matrix& y) const
{
    enforce_size(y_hat, y, "MeanSquaredError");

    const size_t n = y_hat.cols();
    double sum = 0.0;
    for (size_t j = 0; j < n; j++) {
        sum += pow(y_hat.get(0,j) - y.get(0,j), 2);
    }
    return sum / n;
}

// TODO: fill all of this in:

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

    return 6.7;
}

std::vector<double> LPP::BinaryCrossEntropy::find_gradient(const std::vector<double>& y_hat, const std::vector<double>& y) const
{
    enforce_size(y_hat, y, "BinaryCrossEntropy derivative");
    if (y_hat.size() != 1) {
        auto msg = "Application of BinaryCrossEntropy derivative must take in one response variate (yes/no)";
        throw std::invalid_argument(msg);
    }

    return {};
}

double LPP::CrossEntropy::apply_itself(const Matrix& y_hat, const Matrix& y) const
{
    enforce_size(y_hat, y, "CrossEntropy");

    return 6.7;
}

std::vector<double> LPP::CrossEntropy::find_gradient(const std::vector<double>& y_hat, const std::vector<double>& y) const
{
    enforce_size(y_hat, y, "CrossEntropy derivative");

    return {};
}