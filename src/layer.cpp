#include "layer.h"
#include <iostream>

template <typename number>
LPP::Layer<number>::Layer(
    size_t input_size,
    size_t output_size,
    std::shared_ptr<Activation<number>>& initActivationFunction,
    const std::shared_ptr<ProbabilityDistribution>& initProbDistn
) {
    if (initProbDistn) {
        weights__ = std::make_unique<Matrix<number>>(output_size, input_size, initProbDistn);
        biases__ = std::make_unique<Vect<number>>(output_size);
        for (size_t i = 0; i < output_size; i++) {
            (*biases__)[i] = initProbDistn->sample();
        }

    } else {
        // Start with all weights equal to 0 if no distribution specified. This is not recommended in practice
        weights__ = std::make_unique<Matrix<number>>(output_size, input_size);
        biases__ = std::make_unique<Vect<number>>(output_size, 0.0);
    }
    activation_func__ = initActivationFunction;
}

template <typename number>
LPP::Layer<number>::Layer(
    Weights& initWeights,
    Biases&  initBiases,
    std::shared_ptr<Activation<number>>& initActivationFunction
) :
    weights__{std::move(initWeights)},
    biases__{std::move(initBiases)},
    activation_func__{initActivationFunction}
{}
    // weights__ = std::move(given_weights);
    // biases__ = std::move(given_biases);
    // activation_func__ = af;

template <typename number>
void LPP::Layer<number>::apply_activation_layer__(Vect<number>& z) const
{
    for (size_t i = 0; i < z.size(); i++) {
       z[i] = activation_func__->applyActivation(z[i]);
    }
}

template <typename number>
void LPP::Layer<number>::printLayer(std::ostream& os) const {
    os << "Layer weights:\n";
    weights__->displayEntries(os);
    os << "Bias weights:\n";
    weights__->displayElements(os);
    os << "Activation: " << activation_func__->who() << std::endl;
}

// void LPP::print_object(const Layer& l) {l.display();}
