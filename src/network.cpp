#include "network.hpp"
#include "checking/check.hpp"
#include <utility>
#include <fstream>
#include <sstream>

LPP::Network::Network(
    size_t input_size,
    const std::vector<std::pair<size_t,
    std::shared_ptr<Activation>>>& layer_info,
    const std::shared_ptr<ProbabilityDistribution>& pd
) {
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

void LPP::Network::back_propagation_(
    std::vector<std::unique_ptr<Matrix>>& del_W_partial_sum,
    std::vector<std::unique_ptr<std::vector<float>>>& del_b_partial_sum,
    const std::vector<float>& response_variates,
    const std::vector<float>& explanatory_variates
) const {
    // Used to calculate derivatives recursively
    std::vector<float> prev_gradient;
    std::vector<float> current_gradient;

    // Don't use size_t to avoid underflow
    // Loop through layers from last layer back to first layer
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

void LPP::Network::train(
    const Matrix&                   explanatory_variates,
    const Matrix&                   response_variates,
    size_t                          epochs,
    float                           init_learning_rate,
    const std::shared_ptr<Loss>&    loss_ptr,
    int                             sgd_mini_batch_size,
    const std::shared_ptr<regular::Regularizer> regularization_option,  // shorten this somehow
    std::ostream&                   os
)
{
    enforce_condition(explanatory_variates.rows() == response_variates.rows(),
        "Network::train - different number of explanatory and respose variates");
    enforce_condition(explanatory_variates.rows() != 0,
        "Network::train - training data is empty");
    enforce_condition(init_learning_rate > 0.f,
        "Network::train - initial learning rate must be positive");
    enforce_condition(sgd_mini_batch_size <= (int)explanatory_variates.rows(),
        "Network::train -- mini batch size is larger than number of training points");

    // Training info
    size_t  num_training_examples = explanatory_variates.rows();
    float   learning_rate         = init_learning_rate;
            loss_func_            = loss_ptr;
    
    // SGD info
    bool                          perform_sgd   = (sgd_mini_batch_size > 0);
    std::unique_ptr<std::mt19937> shuffler      = nullptr;                                  // Use to shuffle the permutation every epoch
    std::vector<size_t>           permutation = std::vector<size_t>(num_training_examples); // Used to track explanatory variate and response pairs

    // permutation = {0, 1, ..., n-1}
    std::iota(permutation.begin(), permutation.end(), 0);

    if (perform_sgd) {
        shuffler = std::make_unique<std::mt19937>(std::random_device{}());
    } else {
        sgd_mini_batch_size = num_training_examples;
    }

    // How many batches we need to loop through per epoch
    size_t num_batches = num_training_examples / sgd_mini_batch_size;
    if (num_training_examples % sgd_mini_batch_size != 0) num_batches++;

    // response_variates_hat: holds predicted values for epoch
    // del_W: stores derivatives wrt to weights
    // del_b: stores derivaiives wrt to biases
    LPP::Matrix                                         response_variates_hat(explanatory_variates.rows(), explanatory_variates.cols());
    std::vector<std::unique_ptr<Matrix>>                del_W(layers_.size());
    std::vector<std::unique_ptr<std::vector<float>>>    del_b(layers_.size());

    for (size_t cur_epoch = 0; cur_epoch < epochs; cur_epoch++) {
        os << "Epoch " << cur_epoch + 1 << ": " << std::flush; // Flush in case of crash during training
        float current_loss = 0.f;

        // Shuffle training data if using stochastic gradient descent
        if (perform_sgd) std::shuffle(permutation.begin(), permutation.end(), *shuffler);

        // If we are not using sgd, then the below will only trigger once
        for (size_t cur_mini_batch = 0; cur_mini_batch < num_batches; cur_mini_batch++)
        {
            // Initialize derivatives to be filled in by backpropagation
            initialize_gradients_to_zero_(del_W, del_b);

            size_t batch_start_idx   = cur_mini_batch * sgd_mini_batch_size;
            size_t batch_end_idx     = std::min(batch_start_idx + sgd_mini_batch_size, num_training_examples);
            size_t actual_batch_size = batch_end_idx - batch_start_idx;   // If there is remainder

            // Begin to populate derivatives
            // Each item processed corresponds to a (x,y) pair within the loss' sum
            process_training_examples_(
                del_W,
                del_b,
                batch_start_idx,
                batch_end_idx,
                explanatory_variates,
                response_variates,
                response_variates_hat,
                permutation
            );

            // This step changes layer weights and biases!
            update_parameters_(
                del_W,
                del_b,
                actual_batch_size,
                learning_rate,
                regularization_option,
                current_loss
            );
        }
        
        // Compute loss
        current_loss += loss_func_->apply_loss(response_variates_hat, response_variates);
        os << "Loss: " << current_loss << "\n\n";
    }
}

/*
Looping over each training example
Backpropagation will add partial sums to gradients
*/
void LPP::Network::initialize_gradients_to_zero_(
        std::vector<std::unique_ptr<LPP::Matrix>>& delL_delW,
        std::vector<std::unique_ptr<std::vector<float>>>& delL_delb
) const {
    for (size_t cur_layer = 0; cur_layer < layers_.size(); cur_layer++) {
        size_t in     = layers_[cur_layer]->weights_->cols();
        size_t out    = layers_[cur_layer]->weights_->rows();

        delL_delW[cur_layer] = std::make_unique<Matrix>(out, in);
        delL_delb[cur_layer] = std::make_unique<std::vector<float>>(out, 0.0);
    }
}

/*
For all (x_i, y_i) where i in [start, end-1]
- perform forward_propagation by firing the pair through the network
- perform backprogagation by moving back through the layers in reverse order
*/
void LPP::Network::process_training_examples_(
    std::vector<std::unique_ptr<LPP::Matrix>>& delL_delW,
    std::vector<std::unique_ptr<std::vector<float>>>& delL_delb,
    size_t start,
    size_t end,
    const LPP::Matrix& explanatory_variates,
    const LPP::Matrix&  response_variates,
    LPP::Matrix& response_variates_hat,
    const std::vector<size_t>& permutation
) const {
    for (size_t t = start; t < end; t++) {
        size_t idx = permutation[t];

        response_variates_hat[idx] = forward_propagation_(
            explanatory_variates[idx],
            true /* indicates we're training the model */
        );

        back_propagation_(
            delL_delW,
            delL_delb,
            response_variates[idx],
            explanatory_variates[idx]
        );
    }
}

/*
Multiply the derivatives by the learning rate
Divide the derivatives by the batch size because the loss is an average
*/
void LPP::Network::update_parameters_(
    std::vector<std::unique_ptr<Matrix>>& delL_delW,
    std::vector<std::unique_ptr<std::vector<float>>>& delL_delb,
    size_t batch_size,
    float cur_learning_rate,
    const std::shared_ptr<regular::Regularizer> regularization_option,  // shorten this somehow
    float& loss
) {
    for (size_t cur_layer = 0; cur_layer < layers_.size(); cur_layer++) {
        // W <- W - α ∇_W L

        *delL_delW[cur_layer]         *= 1.f / batch_size;      // Divide sum to obtain average
        if (regularization_option) {

            regularization_option->add_regularization_term_derivative(
                *(layers_[cur_layer]->weights_),
                *(delL_delW[cur_layer])
            );

            loss += regularization_option->add_regularization_loss_penalty(*(layers_[cur_layer]->weights_));
        }
        *delL_delW[cur_layer]         *= cur_learning_rate;     // Scale derivative by learning rate
        *layers_[cur_layer]->weights_ -= *delL_delW[cur_layer]; // Subtract for descent step

        // b <- b - α ∇_b L

        *delL_delb[cur_layer]         *= cur_learning_rate / batch_size;
        *layers_[cur_layer]->biases_  -= *delL_delb[cur_layer];
    }
}