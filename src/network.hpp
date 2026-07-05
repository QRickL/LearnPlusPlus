#ifndef LPP_NETWORK_HPP
#define LPP_NETWORK_HPP

#include "layer.hpp"
#include "functions/losses.hpp"
#include "options/training_options.hpp"

namespace LPP {

const std::string MODEL_SAVE_END_MSG = "Check out: https://github.com/QRickL/LearnPlusPlus";

class Network {

    using Vec = std::vector<float>;
    template <typename T>
    using up = std::unique_ptr<T>;
    template <typename T>
    using sp = std::shared_ptr<T>;

    std::vector<up<Layer>> layers_;
    sp<loss::Loss> loss_func_;

    // Fire through the network. Layer outputs, pre and post activation, are saved if training = true
    Vec forward_propagation_(
        Vec current_fire,
        bool training
    ) const;

    // Populates del_W_partial_sum with the derivatives of loss wrt to individual weights
    // Populates del_b_partial_sum with the derivatives of loss wrt to individual biases
    void back_propagation_(
        std::vector<up<Matrix>>& del_W_partial_sum,
        std::vector<up<Vec>>&    del_b_partial_sum,
        const Vec&               response,
        const Vec&               features
    ) const;

    void initialize_gradient_sizes_(
        std::vector<up<Matrix>>& delL_delW_partial_sum,
        std::vector<up<Vec>>& delL_delb_partial_sum
    ) const;

    void initialize_partial_sums_to_zero_(
        std::vector<up<Matrix>>& delL_delW_partial_sum,
        std::vector<up<Vec>>& delL_delb_partial_sum
    ) const;

    void process_training_examples_(
        std::vector<up<Matrix>>&       delL_delW,
        std::vector<up<Vec>>&          delL_delb,
        size_t                         start,
        size_t                         end,
        const Matrix&                  training_features,
        const Matrix&                  training_responses,
        LPP::Matrix&                   estimated_training_responses,
        const up<std::vector<size_t>>& permutation
    ) const;

    void update_parameters_(
        std::vector<up<Matrix>>&        delL_delW,
        std::vector<up<Vec>>&           delL_delb,
        size_t                          batch_size,
        float                           cur_learning_rate,
        const sp<regular::Regularizer>& regularization_option  // shorten this somehow
    );

    float validation_loss_(
        const Matrix& validation_features,
        const Matrix& validation_responses
    ) const;

public:
    // Manually specify model architecture
    // See examples
    Network(
        size_t input_size,
        const std::vector<std::pair<size_t, const activations::Activation*>>& layer_info,
        const sp<distribution::ProbabilityDistribution>& pd = nullptr
    );

    // Load model + weights from file
    Network(const std::string& filepath, std::ostream& os = std::cout);

    // Saving model architecture and weights to a file
    void save_model(const std::string& filepath) const;

    // Single fire through network
    Vec inference(const Vec& x) const;

    // Train model using training set and response variates
    void train(
        const Matrix&               training_features,
        const Matrix&               training_responses,
        size_t                      epochs,
        float                       init_learning_rate,
        const sp<loss::Loss>&       loss_ptr,
        const ExtraTrainingOptions& options = ExtraTrainingOptions()
    );

};

} // namespace LPP

#endif