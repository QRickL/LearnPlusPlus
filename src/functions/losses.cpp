#include "losses.h"
#include <cmath>

LPP::Loss::~Loss() {}

double LPP::MeanSquaredError::apply_itself(const Matrix& y_hat, const Matrix& y)
{
    if (!same_dims(y_hat, y)) {
        const auto msg = "Matrix dimensions for application of MeanSquaredError do not match";
        throw std::invalid_argument(msg);
    }

    // Both inputs should be of dimension 1xn
    // nx1 wastes space
    if (y_hat.rows() != 1 || y.rows() != 1) {
        const auto msg = "Matrices for application of MeanSquaredError have non-1 rows";
        throw std::invalid_argument(msg);
    }

    const size_t n = y_hat.cols();
    double ans = 0.0;
    for (size_t j = 0; j < n; j++) {
        // TODO: fill in lol
    }
    return ans / n;
}