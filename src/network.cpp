#include "network.h"
#include "checking/check.h"
#include <utility>
#include <fstream>
#include <sstream>

LPP::Network::Network(size_t input_size, const std::vector<std::pair<size_t, std::shared_ptr<Activation>>>& layer_info, const std::shared_ptr<ProbabilityDistribution>& pd)
{
    enforce_condition(!layer_info.empty(), "Network::Network - layer_info vector cannot be empty");

    loss_func_ = nullptr; // Loss function will be assigned when training
    size_t in_layer_size;
    size_t out_layer_size = input_size;

    for (const auto& layer : layer_info) {
        in_layer_size = out_layer_size;
        out_layer_size = layer.first;

        layers_.push_back(std::make_unique<Layer>(
            in_layer_size,
            out_layer_size,
            layer.second,
            pd
        ));
    }
}

// TODO: add comments for constructor using filepath
LPP::Network::Network(const std::string& filepath, std::ostream& os)
{
    std::ifstream model_file{filepath};
    enforce_condition(model_file.is_open(), "Network::Network - file failed to open...");
    
    os << "\nLoading neural network from file: " + filepath << '\n';
    std::string cur_line;

    // Verify we are reading a correct file format
    getline(model_file, cur_line);
    enforce_condition(cur_line == "LearnPlusPlus", "Network::Network - not given a LearnPlusPlus file");

    getline(model_file, cur_line);
    enforce_condition(cur_line == "NeuralNetwork", "Network::Network - file type is not neural network");

    loss_func_ = nullptr;
    while (true) {
        // Read in layer number
        model_file >> cur_line;
        if (cur_line == "END") break;
        
        float val;
        size_t out, in;
        model_file >> out >> in;
        os << cur_line << ":\n";
        os << "Input size: " << in << "\nOutput size: " << out << '\n';

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
        enforce_condition(LPP::choose_activation.count(cur_line), "Network::Network - invalid activation function: " + cur_line);

        cur_act = LPP::choose_activation.at(cur_line);
        os << "Activation: " + cur_line << "\n\n";

        // Construct layer and push back
        layers_.push_back(std::make_unique<Layer>(cur_weights, cur_biases, cur_act));
    }
    os << std::endl;
}

// TODO: add commends for saving model
void LPP::Network::save_model(const std::string& filepath) const
{
    std::ofstream model_file{filepath};
    enforce_condition(model_file.is_open(), "Network::save_model - file failed to open...");

    model_file << "LearnPlusPlus\nNeuralNetwork\n";

    for (size_t cur_layer = 0; cur_layer < layers_.size(); cur_layer++) {
        // What layer?
        model_file << "Layer" << cur_layer << '\n';

        // Matrix dimensions
        size_t out = layers_[cur_layer]->weights_->rows();
        size_t in = layers_[cur_layer]->weights_->cols();
        model_file << out << ' ' << in << '\n';

        // Weight contents
        for (size_t i = 0; i < out; i++) {
            for (size_t j = 0; j < in; j++) {
                model_file << (*layers_[cur_layer]->weights_)[i][j] << ' ';
            }
            model_file << '\n';
        }

        // Bias contents
        for (size_t i = 0; i < layers_[cur_layer]->biases_->size(); i++) {
            model_file << (*layers_[cur_layer]->biases_)[i] << ' ';
        }
        model_file << '\n';

        // Activation function
        model_file << layers_[cur_layer]->activation_func_->who() << '\n';
    }
    model_file << "END\n";
    model_file << LPP::MODEL_SAVE_END_MSG << '\n';
    model_file.close();
}

std::vector<float> LPP::Network::forward_propagation_(std::vector<float> current_fire, bool training) const
{
    for (auto& layer : layers_) {
        // z = Wx + b
        current_fire = (*layer->weights_) * current_fire + (*layer->biases_);
        if (training) {
            layer->pre_activation_vals_ = current_fire; // Store if training
        }

        // a = σ(z)
        layer->apply_activation_layer_(current_fire);   // Store if training
        if (training) {
            layer->post_activation_vals_ = current_fire;
        }
    }
    return current_fire;
}

void LPP::Network::back_propagation_(std::vector<std::unique_ptr<Matrix>>& del_W_partial_sum, std::vector<std::unique_ptr<std::vector<float>>>& del_b_partial_sum, const std::vector<float>& response_variates, const std::vector<float>& explanatory_variates) const
{
    // Used to calculate derivatives recursively
    std::vector<float> prev_gradient;
    std::vector<float> current_gradient;

    // Don't use size_t to avoid underflow
    for (int cur_layer = layers_.size() - 1; cur_layer >=0 ; cur_layer--) {
        
        if (cur_layer == layers_.size() - 1) {
            // Looking at last layer, calculate gradient using loss
            current_gradient = loss_func_->find_gradient(layers_[cur_layer]->post_activation_vals_, response_variates);
        }
        else {
            // Looking at non-last layer, calculate gradient recursively
            size_t forward_layer_size = layers_[cur_layer+1]->weights_->rows();
            size_t current_layer_size = layers_[cur_layer]->weights_->rows();
            current_gradient = std::vector<float>(current_layer_size, 0.0);

            for (size_t c = 0; c < current_layer_size; c++) {
                for (size_t k = 0; k < forward_layer_size; k++) {
                    float delL_dela1 = prev_gradient[k];
                    float dela1_delz = layers_[cur_layer+1]->activation_func_->apply_derivative(layers_[cur_layer+1]->pre_activation_vals_[k]);
                    float delz_dela0 = layers_[cur_layer+1]->weights_->get(k,c);

                    // Chain rule: delL_dela0 = delL_dela1 * dela1_delz * delz_dela0
                    current_gradient[c] += delL_dela1 * dela1_delz * delz_dela0;
                }
            }
        }

        for (size_t i = 0; i < layers_[cur_layer]->weights_->rows(); i++) {
            // Define intermediate value to prevent eyesore below
            float imed_value = current_gradient[i] * layers_[cur_layer]->activation_func_->apply_derivative(layers_[cur_layer]->pre_activation_vals_[i]);

            // Update derivatives wrt bias
            (*del_b_partial_sum[cur_layer])[i] += imed_value;

            // Update derivatives wrt weights
            for (size_t j = 0; j < layers_[cur_layer]->weights_->cols(); j++) {
                if (cur_layer > 0) {
                    (*del_W_partial_sum[cur_layer])[i][j] += imed_value * layers_[cur_layer-1]->post_activation_vals_[j];
                } else {
                    (*del_W_partial_sum[cur_layer])[i][j] += imed_value * explanatory_variates[j];
                }
            }
        }
        // The previous gradient is not needed anymore. Set current to previous
        std::swap(current_gradient, prev_gradient);
    }
}

std::vector<float> LPP::Network::inference(const std::vector<float>& x) const
{
    return forward_propagation_(
        x, 
        false /* indicates model not training */
    );
}

float LPP::Network::train(
    const Matrix& explanatory_variates,
    const Matrix& response_variates,
    size_t epochs,
    float init_learning_rate,
    const std::shared_ptr<Loss>& loss_ptr,
    std::ostream& os
)
{
    enforce_condition(explanatory_variates.rows() == response_variates.rows(), "Network::train - different number of explanatory and respose variates");
    enforce_condition(explanatory_variates.rows() != 0, "Network::train - training data is empty");
    enforce_condition(init_learning_rate > 0.f, "Network::train - initial learning rate must be positive");

    size_t num_training_examples = explanatory_variates.rows();
    float learning_rate          = init_learning_rate;
    loss_func_                   = loss_ptr;
    float current_loss;

    // response_variates_hat: holds predicted values for epoch
    // del_W: stores derivatives wrt to weights
    // del_b: stores derivaiives wrt to biases
    LPP::Matrix                                         response_variates_hat(explanatory_variates.rows(), explanatory_variates.cols());
    std::vector<std::unique_ptr<Matrix>>                del_W(layers_.size());
    std::vector<std::unique_ptr<std::vector<float>>>    del_b(layers_.size());

    for (size_t cur_epoch = 0; cur_epoch < epochs; cur_epoch++) {
        os << "Epoch " << cur_epoch + 1 << ": " << std::flush; // Flush in case of crash during training

        // Initialize all gradient sums to 0
        // Gradient sums filled in during back propagation
        for (size_t cur_layer = 0; cur_layer < layers_.size(); cur_layer++) {
            size_t in     = layers_[cur_layer]->weights_->cols();
            size_t out    = layers_[cur_layer]->weights_->rows();

            del_W[cur_layer] = std::make_unique<Matrix>(out, in);
            del_b[cur_layer] = std::make_unique<std::vector<float>>(out, 0.0);
        }

        // Looping over each training example
        // Backpropagation will add partial sums to gradients
        for (size_t t = 0; t < num_training_examples; t++) {
            response_variates_hat[t] = forward_propagation_(
                explanatory_variates[t],
                true /* indicates we're training the model */
            );

            back_propagation_(
                del_W,
                del_b,
                response_variates[t],
                explanatory_variates[t]
            );
        }

        // Update parameters!
        for (size_t cur_layer = 0; cur_layer < layers_.size(); cur_layer++) {
            // W <- W - α ∇_W L
            *del_W[cur_layer]           *= learning_rate / num_training_examples;
            *layers_[cur_layer]->weights_ -= *del_W[cur_layer];

            // b <- b - α ∇_b L
            *del_b[cur_layer]           *= learning_rate / num_training_examples;
            *layers_[cur_layer]->biases_  -= *del_b[cur_layer];
        }
        
        // Compute loss
        current_loss = loss_func_->apply_loss(response_variates_hat, response_variates);
        os << "Loss: " << current_loss << "\n\n";
    }
    return current_loss;
}
