#include "network.hpp"
#include "checking/check.hpp"
#include <utility>
#include <fstream>
#include <sstream>

LPP::Network::Network(
    size_t input_size,
    const std::vector<std::pair<size_t, const activations::Activation*>>& layer_info,
    distribution::ProbabilityDistribution* pd
) {
    enforce_condition(!layer_info.empty(), "Network::Network - layer_info vector cannot be empty");
    enforce_condition(pd, "Network::Network - provided probability distribution is nullptr");

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
        const activations::Activation* cur_act = nullptr;
        enforce_condition(LPP::activations::choose_activation.count(cur_line), "Network::Network - invalid activation function: " + cur_line);

        cur_act = LPP::activations::choose_activation.at(cur_line);
        os << "activations::Activation: " + cur_line << "\n\n";

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

    for (size_t cur_layer_idx = 0; cur_layer_idx < layers_.size(); cur_layer_idx++) {
        // What layer?
        model_file << "Layer" << cur_layer_idx << '\n';
        auto& layer = layers_[cur_layer_idx];

        // Matrix dimensions
        size_t out = layer->weights_->rows();
        size_t in = layer->weights_->cols();
        model_file << out << ' ' << in << '\n';

        // Weight contents
        for (size_t i = 0; i < out; i++) {
            for (size_t j = 0; j < in; j++) {
                model_file << (*layer->weights_)[i][j] << ' ';
            }
            model_file << '\n';
        }

        // Bias contents
        for (size_t i = 0; i < layer->biases_->size(); i++) {
            model_file << (*layer->biases_)[i] << ' ';
        }
        model_file << '\n';

        // activations::Activation function
        model_file << layer->activation_func_->who() << '\n';
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
            layer->pre_activation_vals_ = current_fire; // Store copy 'z' if training
        }

        // a = σ(z)
        layer->activation_func_->apply_activation(current_fire); // current_fire is modified in place
        if (training) {
            layer->post_activation_vals_ = current_fire;
        }
    }
    return current_fire;
}

void LPP::Network::back_propagation_(
    std::vector<std::unique_ptr<Matrix>>& del_W_partial_sum,
    std::vector<std::unique_ptr<std::vector<float>>>& del_b_partial_sum,
    const std::vector<float>& response,
    const std::vector<float>& features
) const {
    // Used to calculate derivatives recursively
    std::vector<float> prev_gradient;
    std::vector<float> current_gradient;

    // Don't use size_t to avoid underflow
    // Loop through layers from last layer back to first layer
    for (int cur_layer_idx = layers_.size() - 1; cur_layer_idx >=0 ; cur_layer_idx--) {
        auto& layer = layers_[cur_layer_idx];
        
        if (cur_layer_idx == layers_.size() - 1) {
            // Looking at last layer, calculate gradient using loss
            current_gradient = loss_func_->find_gradient(layer->post_activation_vals_, response);
        }
        else {
            // 'forward_layer' is the layer which comes after current layer when firing
            // Don't call it 'next_layer' because it could be confused with next layer in the backwards loop
            auto& foward_layer = layers_[cur_layer_idx+1];

            // Looking at non-last layer, calculate gradient recursively
            size_t forward_layer_size = foward_layer->weights_->rows();
            size_t current_layer_size = layer->weights_->rows();
            current_gradient = std::vector<float>(current_layer_size, 0.0);

        // TODO: change apply_derivative
            for (size_t c = 0; c < current_layer_size; c++) {
                for (size_t k = 0; k < forward_layer_size; k++) {
                    float delL_dela1 = prev_gradient[k];
                    float dela1_delz = foward_layer->activation_func_->apply_derivative(foward_layer->pre_activation_vals_[k]);
                    float delz_dela0 = foward_layer->weights_->get(k,c);

                    // Chain rule: delL_dela0 = delL_dela1 * dela1_delz * delz_dela0
                    current_gradient[c] += delL_dela1 * dela1_delz * delz_dela0;
                }
            }
        }

        // TODO: change apply_derivative
        for (size_t i = 0; i < layer->weights_->rows(); i++) {
            // Define intermediate value to prevent eyesore below
            float imed_value = current_gradient[i] * layer->activation_func_->apply_derivative(layer->pre_activation_vals_[i]);

            // Update derivatives wrt bias
            (*del_b_partial_sum[cur_layer_idx])[i] += imed_value;

            // Update derivatives wrt weights
            // Different mode for very first layer
            for (size_t j = 0; j < layer->weights_->cols(); j++) {
                if (cur_layer_idx > 0) {
                    (*del_W_partial_sum[cur_layer_idx])[i][j] += imed_value * layers_[cur_layer_idx-1]->post_activation_vals_[j];
                } else {
                    (*del_W_partial_sum[cur_layer_idx])[i][j] += imed_value * features[j];
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
    const Matrix&                       training_features,
    const Matrix&                       training_responses,
    size_t                              epochs,
    float                               init_learning_rate,
    const loss::Loss*                   loss_ptr,
    const ExtraTrainingOptions&         options
)
{
    enforce_condition(training_features.rows() == training_responses.rows(),
        "Network::train - different number of feature and respose variates");
    enforce_condition(training_features.rows() != 0,
        "Network::train - training data is empty");
    enforce_condition(init_learning_rate > 0.f,
        "Network::train - initial learning rate must be positive");
    enforce_condition(loss_ptr,
        "Network::train - provided loss function is nullptr");
    enforce_condition(!options.use_mini_batch() || options.mini_batch_size() <= training_features.rows(),
        "Network::train - mini batch size is larger than number of training points");
    enforce_condition(!options.use_validation() || options.validation_features().cols() == training_features.cols(),
        "Network::train - training and validation features have different dimenions (matrix column error)");
    enforce_condition(!options.use_validation() || options.validation_responses().cols() == training_responses.cols(),
        "Network::train - training and validation responses have different dimenions (matrix column error)");
    
    // Training info
    size_t  num_training_examples = training_features.rows();
    size_t  mini_batch_size       = num_training_examples;
    float   learning_rate         = init_learning_rate;
            loss_func_            = loss_ptr;
    
    // Mini-batch info for SGD
    std::mt19937 shuffler;                                      // Use to shuffle the permutation every epoch
    std::unique_ptr<std::vector<size_t>> permutation = nullptr; // Used to track explanatory variate and response pairs if applicable

    if (options.use_mini_batch()) {
        shuffler = std::mt19937(std::random_device{}());
        mini_batch_size = options.mini_batch_size();

        permutation = std::make_unique<std::vector<size_t>>(num_training_examples);
        std::iota(permutation->begin(), permutation->end(), 0); // permutation = {0, 1, ..., n-1}
    }

    // How many batches we need to loop through per epoch
    // Add extra batch to train on remainder... this is probably unstable if the remainder is small
    size_t num_batches = num_training_examples / mini_batch_size;
    if (num_training_examples % mini_batch_size != 0) num_batches++;

    // estimated_training_responses: holds predicted values for epoch
    // del_W: stores derivatives wrt to weights
    // del_b: stores derivaiives wrt to biases
    LPP::Matrix                                      estimated_training_responses(training_features.rows(), training_responses.cols());
    std::vector<std::unique_ptr<Matrix>>             del_W(layers_.size());
    std::vector<std::unique_ptr<std::vector<float>>> del_b(layers_.size());
    initialize_gradient_sizes_(del_W, del_b);

    for (size_t cur_epoch = 0; cur_epoch < epochs; cur_epoch++) {
        if (options.has_output_stream()) {
            options.output_stream() << "Epoch " << cur_epoch + 1 << ':' << std::endl; // Flush in case of crash during training
        }

        // Shuffle training data if using stochastic gradient descent
        if (options.use_mini_batch()) std::shuffle(permutation->begin(), permutation->end(), shuffler);

        // If we are not using SGD, then the below will only trigger once
        for (size_t cur_mini_batch = 0; cur_mini_batch < num_batches; cur_mini_batch++)
        {
            // Initialize derivatives to be filled in by backpropagation
            initialize_partial_sums_to_zero_(del_W, del_b);

            size_t batch_start_idx   = cur_mini_batch * mini_batch_size;
            size_t batch_end_idx     = std::min(batch_start_idx + mini_batch_size, num_training_examples);
            size_t actual_batch_size = batch_end_idx - batch_start_idx;   // If there is remainder

            // Begin to populate derivatives
            // Each item processed corresponds to a (x,y) pair within the loss' sum
            process_training_examples_(
                del_W,
                del_b,
                batch_start_idx,
                batch_end_idx,
                training_features,
                training_responses,
                estimated_training_responses,
                permutation
            );

            // This step changes layer weights and biases!
            update_parameters_(
                del_W,
                del_b,
                actual_batch_size,
                learning_rate,
                options.regularizer()
            );
        }
        
        // Compute training data loss
        float training_data_loss = loss_func_->apply_loss(
            estimated_training_responses,
            training_responses
        );

        // Compute penalty loss
        float penalty_loss = 0.f;
        if (options.use_regularization()) {
            for (auto& layer : layers_) {
                penalty_loss += options.regularizer()->add_regularization_loss_penalty(*layer->weights_);
            }
        }

        float training_loss = training_data_loss + penalty_loss;

        // Compute valiation loss if applicable
        float validation_loss;
        if (options.use_validation()) {
            validation_loss = validation_loss_(
                options.validation_features(),
                options.validation_responses()
            );
            validation_loss += penalty_loss;
        }

        if (options.has_output_stream()) {
            auto& os = options.output_stream();
            
            os << "\tTraining Loss: " << training_loss << '\n';
            if (options.use_validation()) os << "\tValidation Loss: " << validation_loss << '\n';
            os << std::endl;
        }
        if (options.has_metadata_stream()) {
            auto& ms = options.metadata_stream();

            ms << cur_epoch + 1 << ' ' << training_loss;
            if (options.use_validation()) ms << ' ' << validation_loss;
            ms << std::endl;
        }
    }
}

/*
Looping over each training example
Backpropagation will add partial sums to gradients
Have this as a separate function to minimize allocations in training hotpath
*/
void LPP::Network::initialize_gradient_sizes_(
    std::vector<std::unique_ptr<LPP::Matrix>>& delL_delW_partial_sum,
    std::vector<std::unique_ptr<std::vector<float>>>& delL_delb_partial_sum
) const {
    for (size_t cur_layer_idx = 0; cur_layer_idx < layers_.size(); cur_layer_idx++) {
        size_t in     = layers_[cur_layer_idx]->weights_->cols();
        size_t out    = layers_[cur_layer_idx]->weights_->rows();

        delL_delW_partial_sum[cur_layer_idx] = std::make_unique<Matrix>(out, in);
        delL_delb_partial_sum[cur_layer_idx] = std::make_unique<std::vector<float>>(out, 0.0);
    }
}

/*
Set all the gradient partial sums to 0 for new training cycle
TODO: could spawn separate threads for this
*/
void LPP::Network::initialize_partial_sums_to_zero_(
    std::vector<std::unique_ptr<LPP::Matrix>>& delL_delW_partial_sum,
    std::vector<std::unique_ptr<std::vector<float>>>& delL_delb_partial_sum
) const {
    for (size_t cur_layer_idx = 0; cur_layer_idx < layers_.size(); cur_layer_idx++) {
        delL_delW_partial_sum[cur_layer_idx]->set_all(0.f);
        set_all(*delL_delb_partial_sum[cur_layer_idx], 0.f);
    }
}

/*
For all (x_i, y_i) where i in [start, end-1]
- perform forward_propagation_ by firing the pair through the network
- perform backprogagation by moving back through the layers in reverse order
*/
void LPP::Network::process_training_examples_(
    std::vector<std::unique_ptr<LPP::Matrix>>& delL_delW,
    std::vector<std::unique_ptr<std::vector<float>>>& delL_delb,
    size_t start,
    size_t end,
    const LPP::Matrix& training_features,
    const LPP::Matrix& training_responses,
    LPP::Matrix& estimated_training_responses,
    const std::unique_ptr<std::vector<size_t>>& permutation
) const {
    for (size_t t = start; t < end; t++) {
        size_t idx = permutation ? (*permutation)[t] : t;

        estimated_training_responses[idx] = forward_propagation_(
            training_features[idx],
            true /* indicates we're training the model */
        );

        back_propagation_(
            delL_delW,
            delL_delb,
            training_responses[idx],
            training_features[idx]
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
    const regular::Regularizer* regularization_option  // shorten this somehow
) {
    for (size_t cur_layer_idx = 0; cur_layer_idx < layers_.size(); cur_layer_idx++) {
        // W <- W - α ∇_W L
        *delL_delW[cur_layer_idx]         *= 1.f / batch_size;      // Divide sum to obtain average
        if (regularization_option)
        {
            regularization_option->add_regularization_term_derivative(
                *(layers_[cur_layer_idx]->weights_),
                *(delL_delW[cur_layer_idx])
            );
        }
        *delL_delW[cur_layer_idx]         *= cur_learning_rate;     // Scale derivative by learning rate
        *layers_[cur_layer_idx]->weights_ -= *delL_delW[cur_layer_idx]; // Subtract for descent step

        // b <- b - α ∇_b L
        *delL_delb[cur_layer_idx]         *= cur_learning_rate / batch_size;
        *layers_[cur_layer_idx]->biases_  -= *delL_delb[cur_layer_idx];
    }
}

float LPP::Network::validation_loss_(
    const LPP::Matrix& validation_features,
    const LPP::Matrix& validation_responses
) const {
    // Calculate estimated response
    LPP::Matrix estimated_validation_responses(validation_responses.rows(), validation_responses.cols());

    for (size_t t = 0; t < validation_responses.rows(); t++)
    {
        estimated_validation_responses[t] = forward_propagation_(
            validation_features[t],
            false /* only final result is needed */
        );
    }

    // Return validation loss
    return loss_func_->apply_loss(estimated_validation_responses, validation_responses);
}
