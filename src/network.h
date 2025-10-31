#ifndef LPP_NETWORK_H
#define LPP_NETWORK_H

#include "matrix.h"
#include "layer.h"
#include "functions/losses.h"
#include <memory>

namespace LPP {

const std::string model_save_end_msg = "END\nCheck out: https://github.com/QRickL/LearnPlusPlus\n";

class Network {
    std::vector<std::unique_ptr<Layer>> layers;

    // Loss function not required for inference, but required for training
    // TODO: This can be passed to the superclass. Modify superclass constructor later
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
        const std::vector<std::pair<size_t, std::shared_ptr<Activation>>> s
    );

    // Load model + weights from file
    Network(const std::string& filepath);

    // Fire through network, will not save intermediate values
    // Not used for traninig
    std::vector<double> inference(const std::vector<double>& x) const;

    // Saving weights to a file
    void save_model(const std::string& filepath) const;

    // TODO: add args. See network.h
    double train(const Matrix& explan_var, const Matrix& response_var, const size_t epochs, const double init_learning_rate, const std::shared_ptr<Loss>& loss_ptr);

};

} // namespace LPP

#endif

// TODO: compile
// TODO: add option to set optimizers etc...
// TODO: activation will be given to layers to choose
// TODO: initial learning rate
// TODO: early stopping
// TODO: loss functions (start with mse and entropy)
// TODO: configure optimizers
// TODO: add batch sizes, stochastic grad descent