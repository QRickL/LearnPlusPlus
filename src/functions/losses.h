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

class Loss {
public:
    virtual double apply_loss(const Matrix& y_hat, const Matrix& y) const = 0;
    virtual std::vector<double> find_gradient(const std::vector<double>& y_hat, const std::vector<double>& y) const = 0;
    void enforce_size(const Matrix& y_hat, const Matrix& y, const std::string& loss_name) const;
    void enforce_size(const std::vector<double>& y, const std::vector<double>& y_hat, const std::string& loss_name) const;

    virtual ~Loss() = 0;
};

class MeanSquaredError : public Loss {
public:
    double apply_loss(const Matrix& y_hat, const Matrix& y) const override;
    std::vector<double> find_gradient(const std::vector<double>& y_hat, const std::vector<double>& y) const override;

    ~MeanSquaredError() {}
};

// need to enforce that there are only two classes (yes / no)
class BinaryCrossEntropy : public Loss {
public:
    double apply_loss(const Matrix& y_hat, const Matrix& y) const override;
    std::vector<double> find_gradient(const std::vector<double>& y_hat, const std::vector<double>& y) const override;

    ~BinaryCrossEntropy() {}
};

class CrossEntropy : public Loss {
public:
    double apply_loss(const Matrix& y_hat, const Matrix& y) const override;
    std::vector<double> find_gradient(const std::vector<double>& y_hat, const std::vector<double>& y) const override;

    ~CrossEntropy() {}
};

const auto MEAN_SQUARED_ERROR   = std::make_shared<MeanSquaredError>();
const auto BINARY_CROSS_ENTROPY = std::make_unique<BinaryCrossEntropy>();
const auto CROSS_ENTROPY        = std::make_shared<CrossEntropy>();

} // namespace LPP

#endif