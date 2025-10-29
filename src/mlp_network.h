#ifndef LPP_MLP_NETWORK_H
#define LPP_MLP_NETWORK_H

#include <memory>
#include <iostream>
#include "mlp_layer.h"
#include "network.h"
#include "matrix.h"

namespace LPP {

class MLP : public Network {
    std::vector<std::unique_ptr<MLP_Layer>> layers;

    // Loss function not required for inference, but required for training
    // TODO: This can be passed to the superclass. Modify superclass constructor later
    std::shared_ptr<Loss> loss_func;

    std::vector<double> forward_propagation(
        std::vector<double> current_fire,
        const bool training
    ) const;

    void back_propagation(
        std::vector<std::unique_ptr<Matrix>>& del_W_partial_sum,
        std::vector<std::unique_ptr<std::vector<double>>>& del_b_partial_sum
    ) const;

    //void back_propagation()

    // calculate gradients

    // backpropagate

    // training = calculate it for everything then update weights. repeat

public:
    // Manually specify model architecture
    MLP(
        const size_t input_size,
        const std::vector<std::pair<size_t, std::shared_ptr<Activation>>> s
    );

    // Load model + weights from file
    MLP(const std::string& filepath);

    // Fire through network, will not save intermediate values
    // Not used for traninig
    std::vector<double> inference(const std::vector<double>& x) const override;

    // Saving weights to a file
    void save_model(const std::string& filepath) const override;

    // TODO: add args. See network.h
    double train(const Matrix& explan_var, const Matrix& response_var, const size_t epochs, const double init_learning_rate) override;

    ~MLP() {}
};

} // namespace LPP

#endif