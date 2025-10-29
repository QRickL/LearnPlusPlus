#include "mlp_network.h"
#include "utility"

// Could I give some of this to network instead????
LPP::MLP::MLP(const size_t input_size, const std::vector<std::pair<size_t, std::shared_ptr<Activation>>> layer_info)
{
    loss_func = nullptr;

    if (layer_info.empty()) {
        throw std::invalid_argument("Attempted to construct MLP with no layers");
    }

    size_t in;
    size_t out = input_size;

    for (const auto& layer : layer_info) {
        in = out;
        out = layer.first;

        layers.push_back(std::make_unique<MLP_Layer>(in, out, layer.second));
    }
}

LPP::MLP::MLP(const std::string& filepath)
{
    loss_func = nullptr;

    std::ifstream model_file{filepath};
    if (!model_file.is_open()) {
        const auto msg = "Failed to read model from file: " + filepath;
        throw std::ios_base::failure(msg);
    }
    std::cout << "\nLoading MultiLayerPerceptron from file: " + filepath << '\n';
    std::string cur_line;

    // Verify the model type
    getline(model_file, cur_line);
    getline(model_file, cur_line);
    if (cur_line != "MLP") {
        const auto msg = "Attempting to assign wrong format file to MLP";
        throw std::invalid_argument(msg);
    }

    while (true) {
        // Read in layer number (unused)
        model_file >> cur_line;
        if (cur_line == "END") break;
        
        size_t out, in;
        double val;
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
        auto cur_biases = std::make_unique<std::vector<double>>(out);
        for (size_t i = 0; i < out; i++) {
            model_file >> val;
            (*cur_biases)[i] = val;
        }

        // Read in activation
        model_file >> cur_line;

        std::shared_ptr<Activation> cur_act;
        try {
            cur_act = LPP::choose_activation.at(cur_line);
        } catch (std::out_of_range e) {
            const auto msg = "Invalid activation function: " + cur_line;
            throw std::invalid_argument(msg);
        }

        std::cout << "Activation: " + cur_line << "\n\n";

        layers.push_back(std::make_unique<MLP_Layer>(cur_weights, cur_biases, cur_act));
    }
    std::cout << std::endl;
}

std::vector<double> LPP::MLP::forward_propagation(std::vector<double> current_fire, const bool training) const
{
    for (auto& layer : layers) {
        // z = Wx + b
        current_fire = (*layer->weights) * current_fire + (*layer->biases);
        if (training) {
            layer->pre_activation = current_fire;
        }

        // a = σ(z)
        current_fire = layer->apply_activation(current_fire);
        if (training) {
            layer->post_activation = current_fire;
        }
    }
    return current_fire;
}

void LPP::MLP::back_propagation(std::vector<std::unique_ptr<Matrix>>& del_W_partial_sum, std::vector<std::unique_ptr<std::vector<double>>>& del_b_partial_sum) const
{
    // Looping backwards
    for (size_t l = layers.size(); l >=0 ; l--) {

    }
}

std::vector<double> LPP::MLP::inference(const std::vector<double>& x) const
{
    return forward_propagation(x, false);
}

void LPP::MLP::save_model(const std::string& filepath) const
{
    std::ofstream model_file{filepath};
    if (!model_file.is_open()) {
        const auto msg = "Failed to save model to file: " + filepath;
        throw std::ios_base::failure(msg);
    }
    model_file << "Modeltype\nMLP\n";

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
                model_file << layers[l]->weights->get(i,j) << ' ';
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
    model_file << LPP::model_save_end_msg;
    model_file.close();
}

double LPP::MLP::train(
    const Matrix& explan_var,
    const Matrix& response_var,
    const size_t epochs,
    const double init_learning_rate
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
        const auto msg = "Learning rate should be a small positive number";
        throw std::invalid_argument(msg);
    }
    if (loss_func == nullptr) {
        const auto msg = "Loss function not set for training";
        throw std::runtime_error(msg);
    }

    // Have this because learning rate may change, eg: adam
    double          learning_rate = init_learning_rate;
    double          loss;
    const size_t    T = explan_var.rows();

    LPP::Matrix                                         response_var_hat(explan_var.rows(), explan_var.cols());
    std::vector<std::unique_ptr<Matrix>>                del_W_sums(layers.size());
    std::vector<std::unique_ptr<std::vector<double>>>   del_b_sums(layers.size());

    for (size_t e = 0; e < epochs; e++) {
        std::cout << "Epoch " << e << ":\n";

        // Initialize all gradient sums to 0
        for (size_t l = 0; l < layers.size(); l++) {
            const size_t in = layers[l]->weights->cols();
            const size_t out = layers[l]->weights->rows();

            del_W_sums[l] = std::make_unique<Matrix>(out, in);
            del_b_sums[l] = std::make_unique<std::vector<double>>(out, 0.0);
        }

        // Looping over each training example
        // Add partial sums to gradients
        for (size_t t = 0; t < T; t++) {
            std::vector<double> inference_result = forward_propagation(explan_var[t], true);
            back_propagation(del_W_sums, del_b_sums);
            
            // Swap lowkey unsafe if inference_result.size() != resonpose_var_hat[t].size()
            std::swap(inference_result, response_var_hat[t]);
        }

        // Calculate gradients
        // TODO: operator overloads for matrix += and vector +=

        // Update weights
        for (size_t l = 0; l < layers.size(); l++) {
            *del_W_sums[l]      *= learning_rate / T;
            *del_b_sums[l]      *= learning_rate / T;

            // W <- W - α ∇_W L
            *layers[l]->weights -= *del_W_sums[l];
            // b <- b - α ∇_b L
            *layers[l]->biases  -= *del_b_sums[l];
        }
        // Compute loss
        loss = loss_func->apply_derivative(response_var_hat, response_var);
        std::cout << "Loss: " << loss << "\n\n";
    }
    // Training will return the final loss
    return loss;
}
