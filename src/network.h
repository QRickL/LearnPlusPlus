#ifndef LPP_NETWORK_H
#define LPP_NETWORK_H

#include "layer.h"
#include "functions/losses.h"

namespace LPP {

const std::string MODEL_SAVE_END_MSG = "Check out: https://github.com/QRickL/LearnPlusPlus";

class Network {
    std::vector<std::unique_ptr<Layer>> layers;
    std::shared_ptr<Loss> loss_func;

    std::vector<double> forward_propagation(
        std::vector<double> current_fire,
        const bool training
    ) const;

    void back_propagation(
        std::vector<std::unique_ptr<Matrix>>& del_W_partial_sum,
        std::vector<std::unique_ptr<std::vector<double>>>& del_b_partial_sum,
        const std::vector<double>& response_var,
        const std::vector<double>& explan_var
    ) const;

public:
    // Manually specify model architecture
    Network(
        const size_t input_size,
        const std::vector<std::pair<size_t, std::shared_ptr<Activation>>> s,
        const std::shared_ptr<ProbabilityDistribution>& pd = nullptr
    );

    // Load model + weights from file
    Network(const std::string& filepath);

    // Saving model architecture and weights to a file
    void save_model(const std::string& filepath) const;

    // Single fire through network
    std::vector<double> inference(const std::vector<double>& x) const;

    // Train model using training set and response variates
    double train(
        const Matrix& explan_var,
        const Matrix& response_var,
        const size_t epochs,
        const double init_learning_rate,
        const std::shared_ptr<Loss>& loss_ptr
    );

};

} // namespace LPP

#endif