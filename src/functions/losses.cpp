#include "losses.h"
#include "../check/check.h"
#include <cmath>

template <typename number>
LPP::Loss<number>::~Loss() {}

template <typename number>
number LPP::MeanSquaredError<number>::applyLoss(const Matrix<number>& y_hat, const Matrix<number>& y) const
{
    _lpp_check_(sameDims(y_hat, y), "MeanSquaredError application recieved incompatible y_hat and y dimensions");

    const size_t m = y_hat.rows();
    const size_t n = y_hat.cols();
    number sum{0.0};

    for (size_t i = 0; i < m; i ++) {
        for (size_t j = 0; j < n; j++) {
            sum += pow(y_hat.get(i,j) - y.get(i,j), 2);
        }
    }
    return sum / m;
}

template <typename number>
LPP::Vect<number> LPP::MeanSquaredError<number>::findGradient(const LPP::Vect<number>& y_hat, const LPP::Vect<number>& y) const
{
    _lpp_check_(sameDims(y_hat, y), "MeanSquaredError derivative recieved incompatible y_hat and y dimensions");

    return 2 * (y_hat - y);
}

template <typename number>
number LPP::BinaryCrossEntropy<number>::applyLoss(const Matrix<number>& y_hat, const Matrix<number>& y) const
{
    _lpp_check_(sameDims(y_hat, y), "BinaryCrossEntropy application recieved incompatible y_hat and y dimensions");


    const size_t m = y_hat.rows();
    const size_t n = y_hat.cols();
    number sum{0.0};

    for (size_t i = 0; i < n; i ++) {
        sum += y.get(i, 0) * std::log(y_hat.get(i, 0)) + (1 - y.get(i, 0)) * std::log(1 - y_hat.get(i, 0));
    }
    return -sum / m;
}

template <typename number>
LPP::Vect<number> LPP::BinaryCrossEntropy<number>::findGradient(const LPP::Vect<number>& y_hat, const LPP::Vect<number>& y) const
{
    _lpp_check_(sameDims(y_hat, y), "BinaryCrossEntropy derivative recieved incompatible y_hat and y dimensions");

    const number l{(1 - y[0]) / (1 - y_hat[0]) - y[0] / y_hat[0]}
    return {l};
    // TODO: this could cause complications
}

template <typename number>
number LPP::CrossEntropy<number>::applyLoss(const Matrix<number>& y_hat, const Matrix<number>& y) const
{
    _lpp_check_(sameDims(y_hat, y), "CrossEntropy application recieved incompatible y_hat and y dimensions");

    const size_t m = y_hat.rows();
    const size_t n = y_hat.cols();
    number sum{0.0};

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            sum += y.get(i,j) * std::log(y_hat.get(i,j));
        }
    }
    return -sum / m;
}

template <typename number>
LPP::Vect<number> LPP::CrossEntropy<number>::findGradient(const LPP::Vect<number>& y_hat, const LPP::Vect<number>& y) const
{
    _lpp_check_(sameDims(y_hat, y), "CrossEntropy derivative recieved incompatible y_hat and y dimensions");

    return -1 * (y / y_hat);
}