#ifndef LPP_NETWORK_H
#define LPP_NETWORK_H

#include <iostream>
#include "layer.hpp"
#include "functions/losses.hpp"

namespace LPP {

const std::string MODEL_SAVE_END_MSG = "Check out: https://github.com/QRickL/LearnPlusPlus";

class Network {
    std::vector<std::unique_ptr<Layer>> layers_;
    std::shared_ptr<Loss> loss_func_;

    // Fire through the network. Layer outputs, pre and post activation, are saved if training = true
    std::vector<float> forward_propagation_(
        std::vector<float> current_fire,
        bool training
    ) const;

    // Populates del_W_partial_sum with the derivatives of loss wrt to individual weights
    // Populates del_b_partial_sum with the derivatives of loss wrt to individual biases
    void back_propagation_(
        std::vector<std::unique_ptr<Matrix>>& del_W_partial_sum,
        std::vector<std::unique_ptr<std::vector<float>>>& del_b_partial_sum,
        const std::vector<float>& response_variates,
        const std::vector<float>& explanatory_variates
    ) const;

    void initialize_gradients_to_zero_(
        std::vector<std::unique_ptr<Matrix>>& delL_delW,
        std::vector<std::unique_ptr<std::vector<float>>>& delL_delb
    );

    void process_training_examples_(
        std::vector<std::unique_ptr<Matrix>>& delL_delW,
        std::vector<std::unique_ptr<std::vector<float>>>& delL_delb,
        size_t start,
        size_t end,
        const Matrix& explanatory_variates,
        const Matrix& response_variates,
        LPP::Matrix& response_variates_hat,
        const std::vector<size_t>& permutation
    );

    void update_parameters_(
        std::vector<std::unique_ptr<Matrix>>& delL_delW,
        std::vector<std::unique_ptr<std::vector<float>>>& delL_delb,
        size_t batch_size,
        float cur_learning_rate
    );

public:
    // Manually specify model architecture
    Network(
        size_t input_size,
        const std::vector<std::pair<size_t, std::shared_ptr<Activation>>>& layer_info,
        const std::shared_ptr<ProbabilityDistribution>& pd = nullptr
    );

    // Load model + weights from file
    Network(const std::string& filepath, std::ostream& os = std::cout);

    // Saving model architecture and weights to a file
    void save_model(const std::string& filepath) const;

    // Single fire through network
    std::vector<float> inference(const std::vector<float>& x) const;

    // Train model using training set and response variates
    float train(
        const Matrix&                   explanatory_variates,
        const Matrix&                   response_variates,
        size_t                          epochs,
        float                           init_learning_rate,
        const std::shared_ptr<Loss>&    loss_ptr,
        int                             sgd_mini_batch_size = -1, // can change this better later
        std::ostream&                   os = std::cout
    );

};

} // namespace LPP

#endif