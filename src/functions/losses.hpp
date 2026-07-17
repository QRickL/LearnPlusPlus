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
protected:
    using Vec = std::vector<float>;
public:
    virtual float apply_loss(const Matrix& y_hat, const Matrix& y) const = 0;
    virtual Vec find_gradient(const Vec& y_hat, const Vec& y) const = 0;
    virtual void find_gradient_then_write(const Vec& y_hat, const Vec& y, Vec& to_write) const = 0;
    virtual ~Loss() = 0;
};

class MeanSquaredError : public Loss {
public:
    float apply_loss(const Matrix& y_hat, const Matrix& y) const override;
    Vec find_gradient(const Vec& y_hat, const Vec& y) const override;
    virtual void find_gradient_then_write(const Vec& y_hat, const Vec& y, Vec& to_write) const override;
    ~MeanSquaredError() {}
};

// need to enforce that there are only two classes (yes / no)
class BinaryCrossEntropy : public Loss {
public:
    float apply_loss(const Matrix& y_hat, const Matrix& y) const override;
    Vec find_gradient(const Vec& y_hat, const Vec& y) const override;
    virtual void find_gradient_then_write(const Vec& y_hat, const Vec& y, Vec& to_write) const override;
    ~BinaryCrossEntropy() {}
};

class CrossEntropy : public Loss {
public:
    float apply_loss(const Matrix& y_hat, const Matrix& y) const override;
    Vec find_gradient(const Vec& y_hat, const Vec& y) const override;
    virtual void find_gradient_then_write(const Vec& y_hat, const Vec& y, Vec& to_write) const override;
    ~CrossEntropy() {}
};

inline const auto mean_squared_error   = MeanSquaredError();
inline const auto binary_cross_entropy = BinaryCrossEntropy();
inline const auto cross_entropy        = CrossEntropy();

inline constexpr float loss_epsilon = 1e-7f;

} // namespace loss

float clamp(float x, float lower, float upper);

} // namespace LPP

#endif