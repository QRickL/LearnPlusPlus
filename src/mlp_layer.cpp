#include "mlp_layer.h"

LPP::MLP_Layer::MLP_Layer(const size_t input_size, const size_t output_size, std::shared_ptr<Activation> af)
{
    weights = std::make_unique<Matrix>(output_size, input_size);
    biases = std::make_unique<std::vector<double>>(output_size, 0.0);
    act_func = af;
}

LPP::MLP_Layer::MLP_Layer(std::unique_ptr<Matrix>& given_weights, std::unique_ptr<std::vector<double>>& given_biases, std::shared_ptr<Activation> af)
{
    weights = std::move(given_weights);
    biases = std::move(given_biases);
    act_func = af;
}

// TODO: parallelize this
// Marked void rather than vector<double> since pre-activation outputs are not needed
void LPP::MLP_Layer::apply_activation(std::vector<double>& x) const
{
    if (act_func == LPP::IDENTITY) return;

    for (size_t i = 0; i < x.size(); i++) {
        x[i] = act_func->apply_itself(x[i]);
    }
}
