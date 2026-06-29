#include "network.h"
#include <utility>
#include <fstream>
#include <sstream>
#include <iostream>

LPP::Network::Network(const size_t input_size, const std::vector<std::pair<size_t, std::shared_ptr<Activation>>>& layer_info, const std::shared_ptr<ProbabilityDistribution>& pd)
{
    if (layer_info.empty()) {
        const auto msg = "Attempted to construct network with no layers";
        throw std::invalid_argument(msg);
    }

    loss_func = nullptr;
    size_t in;
    size_t out = input_size;

    for (const auto& layer : layer_info) {
        in = out;
        out = layer.first;

        layers.push_back(std::make_unique<Layer>(in, out, layer.second, pd));
    }
}

LPP::Network::Network(const std::string& filepath)
{
    std::ifstream model_file{filepath};
    if (!model_file.is_open()) {
        const auto msg = "Failed to read model from file: " + filepath;
        throw std::ios_base::failure(msg);
    }
    std::cout << "\nLoading neural network from file: " + filepath << '\n';
    std::string cur_line;

    // Verify we are reading a correct file format
    getline(model_file, cur_line);
    if (cur_line != "LearnPlusPlus") {
        const auto msg = "Not given a LearnPlusPlus file";
        throw std::ios_base::failure(msg);
    }
    getline(model_file, cur_line);
    if (cur_line != "NeuralNetwork") {
        const auto msg = "File type is not NeuralNetwork";
        throw std::ios_base::failure(msg);
    }

    loss_func = nullptr;
    while (true) {
        // Read in layer number
        model_file >> cur_line;
        if (cur_line == "END") break;
        
        float val;
        size_t out, in;
        model_file >> out >> in;
        std::cout << cur_line << ":\n";
        std::cout << "Input size: " << in << "\nOutput size: " << out << '\n';

        // Read in weights
        auto cur_weights = std::make_unique<Matrix>(out, in);
        for (size_t i = 0; i < out; i++) {
            for (size_t j = 0; j < in; j++) {
                model_file >> val;
                cur_weights->set(i,j,val);
            }
        }
        // Read in biases
        auto cur_biases = std::make_unique<std::vector<float>>(out);
        for (size_t i = 0; i < out; i++) {
            model_file >> val;
            (*cur_biases)[i] = val;
        }

        // Read in activation
        model_file >> cur_line;
        std::shared_ptr<Activation> cur_act;
        try {
            cur_act = LPP::choose_activation.at(cur_line);
        } catch (std::out_of_range oor) {
            const auto msg = "Invalid activation function: " + cur_line;
            throw std::ios_base::failure(msg);
        }
        std::cout << "Activation: " + cur_line << "\n\n";

        // Construct layer and push back
        layers.push_back(std::make_unique<Layer>(cur_weights, cur_biases, cur_act));
    }
    std::cout << std::endl;
}

void LPP::Network::save_model(const std::string& filepath) const
{
    std::ofstream model_file{filepath};
    if (!model_file.is_open()) {
        const auto msg = "Failed to save model to file: " + filepath;
        throw std::ios_base::failure(msg);
    }
    model_file << "LearnPlusPlus\nNeuralNetwork\n";

    for (size_t l = 0; l < layers.size(); l++) {
        // What layer?
        model_file << "Layer" << l << '\n';

        // Matrix dimensions
        const size_t out = layers[l]->weights->rows();
        const size_t in = layers[l]->weights->cols();
        model_file << out << ' ' << in << '\n';

        // Weight contents
        for (size_t i = 0; i < out; i++) {
            for (size_t j = 0; j < in; j++) {
                model_file << (*layers[l]->weights)[i][j] << ' ';
            }
            model_file << '\n';
        }

        // Bias contents
        for (size_t i = 0; i < layers[l]->biases->size(); i++) {
            model_file << (*layers[l]->biases)[i] << ' ';
        }
        model_file << '\n';

        // Activation function
        model_file << layers[l]->act_func->who() << '\n';
    }
    model_file << "END\n";
    model_file << LPP::MODEL_SAVE_END_MSG << '\n';
    model_file.close();
}

std::vector<float> LPP::Network::forward_propagation(std::vector<float> current_fire, const bool training) const
{
    for (auto& layer : layers) {
        // z = Wx + b
        current_fire = (*layer->weights) * current_fire + (*layer->biases);
        if (training) {
            layer->pre_activation = current_fire;
        }

        // a = σ(z)
        layer->apply_activation(current_fire);
        if (training) {
            layer->post_activation = current_fire;
        }
    }
    return current_fire;
}

void LPP::Network::back_propagation(std::vector<std::unique_ptr<Matrix>>& del_W_partial_sum, std::vector<std::unique_ptr<std::vector<float>>>& del_b_partial_sum, const std::vector<float>& response_var, const std::vector<float>& explan_var) const
{
    // Used to calculate derivatives recursively
    std::vector<float> prev_gradient;
    std::vector<float> current_gradient;

    // Don't use size_t to avoid underflow
    for (int l = layers.size() - 1; l >=0 ; l--) {
        
        if (l == layers.size() - 1) {
            // Looking at last layer, calculate gradient using loss
            current_gradient = loss_func->find_gradient(layers[l]->post_activation, response_var);
        }
        else {
            // Looking at non-last layer, calculate gradient recursively
            const size_t forward_layer_size = layers[l+1]->weights->rows();
            const size_t current_layer_size = layers[l]->weights->rows();
            current_gradient = std::vector<float>(current_layer_size, 0.0);

            for (size_t c = 0; c < current_layer_size; c++) {
                for (size_t k = 0; k < forward_layer_size; k++) {
                    const float delL_dela1 = prev_gradient[k];
                    const float dela1_delz = layers[l+1]->act_func->apply_derivative(layers[l+1]->pre_activation[k]);
                    const float delz_dela0 = layers[l+1]->weights->get(k,c);

                    // Chain rule: delL_dela0 = delL_dela1 * dela1_delz * delz_dela0
                    current_gradient[c] += delL_dela1 * dela1_delz * delz_dela0;
                }
            }
        }

        for (size_t i = 0; i < layers[l]->weights->rows(); i++) {

            const float imed = current_gradient[i] * layers[l]->act_func->apply_derivative(layers[l]->pre_activation[i]);

            // Update derivatives wrt bias
            (*del_b_partial_sum[l])[i] += imed;
            // Update derivatives wrt weights
            for (size_t j = 0; j < layers[l]->weights->cols(); j++) {
                if (l > 0) {
                    (*del_W_partial_sum[l])[i][j] += imed * layers[l-1]->post_activation[j];
                } else {
                    (*del_W_partial_sum[l])[i][j] += imed * explan_var[j];
                }
            }
        }
        std::swap(current_gradient, prev_gradient);
    }
}

std::vector<float> LPP::Network::inference(const std::vector<float>& x) const
{
    return forward_propagation(x, false);
}

float LPP::Network::train(
    const Matrix& explan_var,
    const Matrix& response_var,
    const size_t epochs,
    const float init_learning_rate,
    const std::shared_ptr<Loss>& loss_ptr
)
{
    if (explan_var.rows() != response_var.rows()) {
        const auto msg = "Different number of explanatory and respose variates";
        throw std::invalid_argument(msg);
    }
    if (explan_var.rows() == 0) {
        const auto msg = "Training data is empty!";
        throw std::invalid_argument(msg);
    }
    if (init_learning_rate <= 0) {
        const auto msg = "Learning rate should be positive";
        throw std::invalid_argument(msg);
    }

    const size_t num_training_examples = explan_var.rows();
    float learning_rate = init_learning_rate;
    loss_func = loss_ptr;
    float loss;

    // response_var_hat: holds predicted values for epoch
    // del_W: stores derivatives wrt to weights
    // del_b: stores derivaiives wrt to biases
    LPP::Matrix                                         response_var_hat(explan_var.rows(), explan_var.cols());
    std::vector<std::unique_ptr<Matrix>>                del_W(layers.size());
    std::vector<std::unique_ptr<std::vector<float>>>   del_b(layers.size());

    for (size_t cur_epoch = 0; cur_epoch < epochs; cur_epoch++) {
        std::cout << "Training epoch " << cur_epoch + 1 << ":\n";

        // Initialize all gradient sums to 0
        for (size_t l = 0; l < layers.size(); l++) {
            const size_t in     = layers[l]->weights->cols();
            const size_t out    = layers[l]->weights->rows();

            del_W[l] = std::make_unique<Matrix>(out, in);
            del_b[l] = std::make_unique<std::vector<float>>(out, 0.0);
        }
        // Looping over each training example
        // Backpropagation will add partial sums to gradients
        for (size_t t = 0; t < num_training_examples; t++) {

            response_var_hat[t] = forward_propagation(explan_var[t], true);
            back_propagation(del_W, del_b, response_var[t], explan_var[t]);

        }
        // Update parameters!
        for (size_t l = 0; l < layers.size(); l++) {
            // W <- W - α ∇_W L
            *del_W[l]           *= learning_rate / num_training_examples;
            *layers[l]->weights -= *del_W[l];

            // b <- b - α ∇_b L
            *del_b[l]           *= learning_rate / num_training_examples;
            *layers[l]->biases  -= *del_b[l];
        }
        // Compute loss
        loss = loss_func->apply_loss(response_var_hat, response_var);
        std::cout << "Loss: " << loss << "\n\n";
    }
    return loss;
}
