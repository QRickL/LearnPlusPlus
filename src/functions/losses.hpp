#ifndef LPP_LOSSES_H
#define LPP_LOSSES_H

#include "../matrix.hpp"
#include <memory>

// Loss functions are used by the network class to perform training
// Available loss functions
// * Mean squared error     - regression tasks
// * Binary cross entropy   - classification tasks
// * Cross entropy          - classification tasks

namespace LPP
{

class Loss {
public:
    virtual float apply_loss(const Matrix& y_hat, const Matrix& y) const = 0;
    virtual std::vector<float> find_gradient(const std::vector<float>& y_hat, const std::vector<float>& y) const = 0;
    virtual ~Loss() = 0;
};

class MeanSquaredError : public Loss {
public:
    float apply_loss(const Matrix& y_hat, const Matrix& y) const override;
    std::vector<float> find_gradient(const std::vector<float>& y_hat, const std::vector<float>& y) const override;
    ~MeanSquaredError() {}
};

// need to enforce that there are only two classes (yes / no)
class BinaryCrossEntropy : public Loss {
public:
    float apply_loss(const Matrix& y_hat, const Matrix& y) const override;
    std::vector<float> find_gradient(const std::vector<float>& y_hat, const std::vector<float>& y) const override;
    ~BinaryCrossEntropy() {}
};

class CrossEntropy : public Loss {
public:
    float apply_loss(const Matrix& y_hat, const Matrix& y) const override;
    std::vector<float> find_gradient(const std::vector<float>& y_hat, const std::vector<float>& y) const override;
    ~CrossEntropy() {}
};

namespace losses {

const auto MEAN_SQUARED_ERROR   = std::make_shared<MeanSquaredError>();
const auto BINARY_CROSS_ENTROPY = std::make_unique<BinaryCrossEntropy>();
const auto CROSS_ENTROPY        = std::make_shared<CrossEntropy>();

}

} // namespace LPP

#endif