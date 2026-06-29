# pragma once

#include "layer.h"
#include "functions/losses.h"
#include <iostream>

namespace LPP {

/*
Main class of LPP which gives you the neural network!

Training data should be of type std::vector<std::vector<float>>
Inference input should be of type std::vector<float>
Inferece output will be std::vector<float>

Even though internally LPP uses a templated custom vector to save space, we still interact with LPP using std::vector
The purpose of using the custom 'number' type is to support quantization and autograd (...in a future update)
*/
template <typename number>
class Network {

    using Weights       = std::unique_ptr<Matrix<number>>;
    using Biases        = std::unique_ptr<Vect<number>>;
    using ActivationPtr = std::shared_ptr<Activation<number>>;
    using LossPtr       = std::shared_ptr<Loss<number>>;
    using ProbDistnPtr  = std::shared_ptr<ProbabilityDistribution>;
    using Layers        = std::vector<std::unique_ptr<Layer<number>>>;
    using LayerInfo     = std::pair<unsigned int, ActivationPtr>;

    Layers layers__;
    LossPtr loss_func__;

    Vect<number> forward_propagation__(
        Vect<number> current_fire__,
        bool training__
    ) const;

    void back_propagation__(
        std::vector<Weights>&       del_W_partial_sum__,
        std::vector<Biases>&        del_b_partial_sum__,
        const std::vector<double>&  response_var__, // what to do with this one...
        const std::vector<double>&  explan_var__
    ) const;

    const std::string model_save_end_msg__ = "Check out: https://github.com/QRickL/LearnPlusPlus";

public:
    // Manually specify model architecture
    Network(
        size_t inputSize,
        const std::vector<LayerInfo>& layerInfos,
        ProbDistnPtr probDistn = nullptr
    );

    // Load model + weights from file
    Network(const std::string& filepath);

    // Saving model architecture and weights to a file
    void saveModel(const std::string& filepath) const;

    // Train model using training set and response variates.
    // sdf
    double train(
        const std::vector<std::vector<float>>& explan_var,
        const std::vector<std::vector<float>>& response_var,
        float init_learning_rate,
        size_t epochs,
        LossPtr& loss_ptr
    );

    std::vector<float> inference(const std::vector<float>& x) const; // Single fire through network. Float version
    std::vector<float> inference(const Vect<number>& x) const;       // Single fire through network. LPP Vect version
};

} // namespace LPP