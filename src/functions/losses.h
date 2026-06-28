#ifndef LPP_LOSSES_H
#define LPP_LOSSES_H

#include "../matrix.h"
#include <memory>

// Loss functions are used by the network class to perform training
// Available loss functions
// * Mean squared error     - regression tasks
// * Binary cross entropy   - classification tasks
// * Cross entropy          - classification tasks

namespace LPP
{

template <typename number>
class Loss {
public:
    virtual number applyLoss(const Matrix<number>& y_hat, const Matrix<number>& y) const = 0;
    virtual Vect<number> findGradient(const Vect<number>& y_hat, const Vect<number>& y) const = 0;

    virtual ~Loss() = 0;
};

template <typename number>
class MeanSquaredError : public Loss<number> {
public:
    number applyLoss(const Matrix<number>& y_hat, const Matrix<number>& y) const override;
    Vect<number> findGradient(const Vect<number>& y_hat, const Vect<number>& y) const override;

    ~MeanSquaredError() {}
};

// TODO: need to enforce that there are only two classes (yes / no)

template <typename number>
class BinaryCrossEntropy : public Loss<number> {
public:
    number applyLoss(const Matrix<number>& y_hat, const Matrix<number>& y) const override;
    Vect<number> findGradient(const Vect<number>& y_hat, const Vect<number>& y) const override;

    ~BinaryCrossEntropy() {}
};

template <typename number>
class CrossEntropy : public Loss<number> {
public:
    number applyLoss(const Matrix<number>& y_hat, const Matrix<number>& y) const override;
    Vect<number> findGradient(const Vect<number>& y_hat, const Vect<number>& y) const override;

    ~CrossEntropy() {}
};

// TODO: resolve thiss
// const auto MEAN_SQUARED_ERROR   = std::make_shared<MeanSquaredError>();
// const auto BINARY_CROSS_ENTROPY = std::make_shared<BinaryCrossEntropy>();
// const auto CROSS_ENTROPY        = std::make_shared<CrossEntropy>();

} // namespace LPP

#endif