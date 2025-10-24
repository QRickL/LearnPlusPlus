#ifndef LPP_LOSSES_H
#define LPP_LOSSES_H

#include "matrix.h"

namespace LPP
{

class Loss {
public:
    virtual double apply_itself(const Matrix& y_hat, const Matrix& y);
    virtual double apply_derivative(const Matrix& y_hat, const Matrix& y);

    // Loss is virtual
    virtual ~Loss() = 0;
};

class MeanSquaredError : public Loss {
public:
    double apply_itself(const Matrix& y_hat, const Matrix& y) override;
    double apply_derivative(const Matrix& y_hat, const Matrix& y) override;

    ~MeanSquaredError();
};

class BinaryCrossEntropy : public Loss {
public:
    double apply_itself(const Matrix& y_hat, const Matrix& y) override;
    double apply_derivative(const Matrix& y_hat, const Matrix& y) override;

    ~BinaryCrossEntropy();
};

class CrossEntropy : public Loss {
public:
    double apply_itself(const Matrix& y_hat, const Matrix& y) override;
    double apply_derivative(const Matrix& y_hat, const Matrix& y) override;

    ~CrossEntropy();
};

} // namespace LPP

#endif