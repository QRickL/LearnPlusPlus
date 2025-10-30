#include "losses.h"
#include <cmath>

LPP::Loss::~Loss() {}

double LPP::MeanSquaredError::apply_itself(const Matrix& y_hat, const Matrix& y)
{
    if (!same_dims(y_hat, y)) {
        const auto msg = "Matrix dimensions for application of MeanSquaredError do not match";
        throw std::invalid_argument(msg);
    }

    const size_t n = y_hat.cols();
    double sum = 0.0;
    for (size_t j = 0; j < n; j++) {
        sum += pow(y_hat.get(0,j) - y.get(0,j), 2);
    }
    return sum / n;
}

// why is this even needed...??? derivative wrt to what. figure out later
double LPP::MeanSquaredError::apply_derivative(const Matrix& y_hat, const Matrix& y)
{
    return 1.0;
}