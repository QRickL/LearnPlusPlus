#ifndef LPP_LOSSES_HPP
#define LPP_LOSSES_HPP

#include "../mathobj/matrix.hpp"
#include <memory>

// Loss functions are used by the network class to perform training
// Available loss functions
// * Mean squared error     - regression tasks
// * Binary cross entropy   - classification tasks
// * Cross entropy          - classification tasks

namespace LPP {
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

inline const auto mean_squared_error   = MeanSquaredError();
inline const auto binary_cross_entropy = BinaryCrossEntropy();
inline const auto cross_entropy        = CrossEntropy();

inline const float loss_epsilon = 1e-7f;

} // namespace loss

float clamp(float x, float lower, float upper);

} // namespace LPP

#endif