#ifndef LPP_NETWORK_H
#define LPP_NETWORK_H

#include "layer.h"
#include "functions/losses.h"

namespace LPP {

const std::string MODEL_SAVE_END_MSG = "Check out: https://github.com/QRickL/LearnPlusPlus";

class Network {
    std::vector<std::unique_ptr<Layer>> layers;
    std::shared_ptr<Loss> loss_func;

    std::vector<float> forward_propagation(
        std::vector<float> current_fire,
        bool training
    ) const;

    void back_propagation(
        std::vector<std::unique_ptr<Matrix>>& del_W_partial_sum,
        std::vector<std::unique_ptr<std::vector<float>>>& del_b_partial_sum,
        const std::vector<float>& response_var,
        const std::vector<float>& explan_var
    ) const;

public:
    // Manually specify model architecture
    Network(
        size_t input_size,
        const std::vector<std::pair<size_t, std::shared_ptr<Activation>>>& layer_info,
        const std::shared_ptr<ProbabilityDistribution>& pd = nullptr
    );

    // Load model + weights from file
    Network(const std::string& filepath);

    // Saving model architecture and weights to a file
    void save_model(const std::string& filepath) const;

    // Single fire through network
    std::vector<float> inference(const std::vector<float>& x) const;

    // Train model using training set and response variates
    float train(
        const Matrix& explan_var,
        const Matrix& response_var,
        size_t epochs,
        float init_learning_rate,
        const std::shared_ptr<Loss>& loss_ptr
    );

};

} // namespace LPP

#endif