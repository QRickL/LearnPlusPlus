#ifndef LPP_LOSSES_H
#define LPP_LOSSES_H

#include "../mathobj/matrix.hpp"
#include <memory>

// Loss functions are used by the network class to perform training
// Available loss functions
// * Mean squared error     - regression tasks
// * Binary cross entropy   - classification tasks
// * Cross entropy          - classification tasks

namespace LPP
{
namespace loss {

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

const auto mean_squared_error   = std::make_shared<MeanSquaredError>();
const auto binary_cross_entropy = std::make_unique<BinaryCrossEntropy>();
const auto cross_entropy        = std::make_shared<CrossEntropy>();

} // namespace loss
} // namespace LPP

#endif