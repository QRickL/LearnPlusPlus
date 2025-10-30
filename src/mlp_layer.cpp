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
std::vector<double> LPP::MLP_Layer::apply_activation(const std::vector<double>& z) const
{
    std::vector<double> a(z.size());

    for (size_t i = 0; i < a.size(); i++) {
        a[i] = act_func->apply_itself(z[i]);
    }
    return a;
}
