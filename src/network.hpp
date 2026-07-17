#ifndef LPP_NETWORK_HPP
#define LPP_NETWORK_HPP

#include "layer.hpp"
#include "functions/losses.hpp"
#include "options/training_options.hpp"

namespace LPP {

const std::string MODEL_SAVE_END_MSG = "Check out: https://github.com/QRickL/LearnPlusPlus";

class Network {
    template <typename T>
    using up = std::unique_ptr<T>;
    using Vec = std::vector<float>;

    std::vector<Layer> layers_;

    // For training
    const loss::Loss* loss_func_;
    mutable size_t max_layer_size_;
    mutable Vec current_fire_buffer_;
    mutable Vec prev_gradient_;
    mutable Vec current_gradient_;
    mutable Matrix prev_jacobian_;
    mutable Matrix current_jacobian_;
    mutable Vec X_i_;
    mutable Vec Y_i_;
    mutable Matrix estimated_validation_responses_;

    // Fire through the network. Layer outputs, pre and post activation, are saved if training = true
    Vec forward_propagation_(
        Vec current_fire,
        bool training
    ) const;

    // Populates del_W_partial_sum with the derivatives of loss wrt to individual weights
    // Populates del_b_partial_sum with the derivatives of loss wrt to individual biases
    void back_propagation_(
        std::vector<Matrix>& del_W_partial_sum,
        std::vector<Vec>&    del_b_partial_sum,
        const Vec&           response,
        const Vec&           features
    ) const;

    void initialize_gradient_sizes_(
        std::vector<Matrix>& delL_delW_partial_sum,
        std::vector<Vec>& delL_delb_partial_sum
    ) const;

    void initialize_partial_sums_to_zero_(
        std::vector<Matrix>& delL_delW_partial_sum,
        std::vector<Vec>& delL_delb_partial_sum
    ) const;

    void process_training_examples_(
        std::vector<Matrix>&           delL_delW,
        std::vector<Vec>&              delL_delb,
        size_t                         start,
        size_t                         end,
        const Matrix&                  training_features,
        const Matrix&                  training_responses,
        LPP::Matrix&                   estimated_training_responses,
        const up<std::vector<size_t>>& permutation
    ) const;

    void update_parameters_(
        std::vector<Matrix>&        delL_delW,
        std::vector<Vec>&           delL_delb,
        size_t                      batch_size,
        float                       cur_learning_rate,
        const regular::Regularizer* regularization_option  // shorten this somehow
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
        const std::vector<std::tuple<
            size_t,
            const activations::Activation*,
            distribution::ProbabilityDistribution*
        >>& layer_info
    );

    // Load model + weights from file
    Network(const std::string& filepath, std::ostream& os = std::cout);

    // Saving model architecture and weights to a file
    void save_model(const std::string& filepath) const;

    // Single fire through network
    Vec inference(const Vec& x) const;
    Vec inference(const std::span<const float> x) const;

    // Train model using training set and response variates
    void train(
        const Matrix&               training_features,
        const Matrix&               training_responses,
        size_t                      epochs,
        float                       init_learning_rate,
        const loss::Loss*           loss_ptr,
        const ExtraTrainingOptions& options = ExtraTrainingOptions()
    );

};

} // namespace LPP

#endif