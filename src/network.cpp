#include "network.hpp"
#include "checking/check.hpp"
#include "options/classify.hpp"
#include <utility>
#include <fstream>
#include <sstream>
#include <algorithm>

LPP::Network::Network(
    size_t input_size,
    const std::vector< std::tuple<size_t, const activations::Activation*, distribution::ProbabilityDistribution*> >& layer_info
) : max_layer_size_{0}, loss_func_{nullptr} // Loss function will be assigned when training
{
    enforce_condition(!layer_info.empty(), "Network::Network - layer_info vector cannot be empty");

    size_t in_layer_size;
    size_t out_layer_size = input_size;

    for (const auto& layer : layer_info) {
        in_layer_size = out_layer_size;
        out_layer_size = std::get<0>(layer);

        if (max_layer_size_ < out_layer_size) max_layer_size_ = out_layer_size;

        layers_.emplace_back(
            in_layer_size,
            out_layer_size,
            std::get<1>(layer),
            std::get<2>(layer)
        );
    }
    current_fire_buffer_.resize(max_layer_size_);
    current_gradient_.resize(max_layer_size_);
    prev_gradient_.resize(max_layer_size_);
    current_jacobian_.resize(max_layer_size_, max_layer_size_);
    prev_jacobian_.resize(max_layer_size_, max_layer_size_);
}

// TODO: add comments for constructor using filepath
LPP::Network::Network(const std::string& filepath, std::ostream& os)
    : max_layer_size_{0}, loss_func_{nullptr}
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

    while (true) {
        // Read in layer number
        model_file >> cur_line;
        if (cur_line == "END") break;
        
        float val;
        size_t out, in;
        model_file >> out >> in;
        os << cur_line << ":\n";
        os << "Input size: " << in << "\nOutput size: " << out << '\n';

        if (max_layer_size_ < out) max_layer_size_ = out; 

        // Read in weights
        auto cur_weights = Matrix(out, in);
        for (size_t i = 0; i < out; i++) {
            for (size_t j = 0; j < in; j++) {
                model_file >> val;
                cur_weights.set(i,j,val);
            }
        }
        // Read in biases
        auto cur_biases = std::vector<float>(out);
        for (size_t i = 0; i < out; i++) {
            model_file >> val;
            cur_biases[i] = val;
        }

        // Read in activation
        model_file >> cur_line;
        const activations::Activation* cur_act = nullptr;
        enforce_condition(LPP::activations::choose_activation.count(cur_line),
            "Network::Network - invalid activation function: " + cur_line);

        cur_act = LPP::activations::choose_activation.at(cur_line);
        os << "Activation function: " + cur_line << "\n\n";

        // Construct layer and push back
        layers_.emplace_back(cur_weights, cur_biases, cur_act);
    }
    os << std::endl;
    current_fire_buffer_.resize(max_layer_size_);
    current_gradient_.resize(max_layer_size_);
    prev_gradient_.resize(max_layer_size_);
    current_jacobian_.resize(max_layer_size_, max_layer_size_);
    prev_jacobian_.resize(max_layer_size_, max_layer_size_);
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
        size_t out = layer.weights_.rows();
        size_t in = layer.weights_.cols();
        model_file << out << ' ' << in << '\n';

        // Weight contents
        for (size_t i = 0; i < out; i++) {
            for (size_t j = 0; j < in; j++) {
                model_file << layer.weights_[i][j] << ' ';
            }
            model_file << '\n';
        }

        // Bias contents
        for (size_t i = 0; i < layer.biases_.size(); i++) {
            model_file << layer.biases_[i] << ' ';
        }
        model_file << '\n';

        // activations::Activation function
        model_file << layer.activation_func_->who() << '\n';
    }
    model_file << "END\n";
    model_file << LPP::MODEL_SAVE_END_MSG << '\n';
    model_file.close();
}

std::vector<float> LPP::Network::forward_propagation_(std::vector<float> current_fire, bool training) const
{
    for (auto& layer : layers_) {
        // z = Wx + b
        /*
        Reason we do mult_add_write instead of:
        current_fire = layer.weights_ * current_fire + layer.biases_;
        is to avoid constructing a vector here during training
        */
        current_fire_buffer_.resize(layer.weights_.rows());
        layer.weights_.mult_add_write(current_fire, layer.biases_, current_fire_buffer_);
        std::swap(current_fire, current_fire_buffer_);

        if (training) {
            // Store copy 'z' if training
            std::copy(current_fire.begin(), current_fire.end(),
                layer.pre_activation_vals_.begin());
        }

        // a = σ(z)
        layer.activation_func_->apply_activation(current_fire); // current_fire is modified in place
        if (training) {
            // Store copy 'a' if training
            std::copy(current_fire.begin(), current_fire.end(),
                layer.post_activation_vals_.begin());
        }
    }
    return current_fire;
}

void LPP::Network::back_propagation_(
    std::vector<Matrix>& del_W_partial_sum,
    std::vector<std::vector<float>>& del_b_partial_sum,
    const std::vector<float>& response,
    const std::vector<float>& features
) const {
    /*
    Used to calculate derivatives recursively:
        prev_gradient_
        current_gradient_
        prev_jacobian_
        current_jacobian_
    They are allocated during network construction to avoid allocations during backprop
    */

    // Don't use size_t to avoid underflow
    // Loop through layers from last layer back to first layer
    for (int cur_layer_idx = layers_.size() - 1; cur_layer_idx >=0 ; cur_layer_idx--) {
        auto& layer = layers_[cur_layer_idx];
        size_t current_layer_size = layer.weights_.rows();
        
        if (cur_layer_idx == layers_.size() - 1) {
            // Looking at last layer, calculate gradient using loss
            // Differentiate directly

            current_gradient_.resize(response.size());
            loss_func_->find_gradient_then_write(layer.post_activation_vals_, response, current_gradient_);

            /*
            Performs this without allocation:
            current_gradient_ = loss_func_->find_gradient(layer.post_activation_vals_, response);
            */
        }
        else {
            // 'forward_layer' is the layer which comes after current layer when firing
            // Don't call it 'next_layer' because it could be confused with next layer in the backwards loop
            auto& forward_layer = layers_[cur_layer_idx+1];
            const size_t forward_layer_size = forward_layer.weights_.rows();
            const bool forward_layer_activation_elements_non_interdependent = forward_layer.activation_func_->elements_non_interdependent();

            // Looking at non-last layer, calculate gradient recursively
            // current_gradient stores the derivative of loss wrt activation in current layer
            // prev_gradient stores the derivative of loss wrt activation in forward layer

            // derivatives.pdf: del L / del a_i
            current_gradient_.assign(current_layer_size, 0.f);

            // Computing derivative of loss wrt i-th activation
            for (size_t i = 0; i < current_layer_size; i++) {
                for (size_t k = 0; k < forward_layer_size; k++) {
                    
                    if (forward_layer_activation_elements_non_interdependent)
                    {
                        float delL_dela1 = prev_gradient_[k];
                        float dela1_delz = forward_layer.activation_func_->apply_derivative(forward_layer.pre_activation_vals_[k]);
                        float delz_dela0 = forward_layer.weights_[k][i];
                        
                        // Chain rule: delL_dela0 = delL_dela1 * dela1_delz * delz_dela0
                        current_gradient_[i] += delL_dela1 * dela1_delz * delz_dela0;
                    }
                    else
                    {
                        float delL_dela1 = prev_gradient_[k];
                        float imed_val = 0.f;

                        for (size_t c = 0; c < forward_layer_size; c++)
                        {
                            float dela1_k_delz_c = prev_jacobian_[k][c];
                            imed_val += dela1_k_delz_c * forward_layer.weights_[c][i];
                        }
                        current_gradient_[i] += delL_dela1 * imed_val;
                    }
                }
            }
        }

        const bool layer_activation_elements_non_interdependent = layer.activation_func_->elements_non_interdependent();
        // Calculate jacobian for this layer's loss wrt activation
        if (!layer_activation_elements_non_interdependent) {
            current_jacobian_.resize_and_set(current_layer_size, current_layer_size, 0.f);
            layer.activation_func_->calculate_jacobian(layer.post_activation_vals_, current_jacobian_);
        }

        // Loop over (i) will hit every bias in layer
        // Loop over (i,j) will hit every weight in layer
        for (size_t i = 0; i < current_layer_size; i++) {

            // See general_activations.pdf for mathematical details
            // Define intermediate value to prevent eyesore when actually adding to partial sums
            float imed_value = 0.f;
            if (layer_activation_elements_non_interdependent)
            {
                //float dela0_i_delz_i = layer->activation_func_->apply_derivative(layer->pre_activation_vals_[i]);
                float dela0_i_delz_i = layer.activation_func_->apply_derivative(layer.pre_activation_vals_[i]);
                imed_value = current_gradient_[i] * dela0_i_delz_i;
            }
            else
            {
                for (size_t c = 0; c < current_layer_size; c++)
                {
                    //float dela0_c_delz_i = layer->activation_func_->jacobian(layer->pre_activation_vals_, c, i);
                    float dela0_c_delz_i = current_jacobian_[c][i];
                    imed_value += current_gradient_[c] * dela0_c_delz_i;
                }
            }

            // Update derivatives wrt bias
            (del_b_partial_sum[cur_layer_idx])[i] += imed_value;

            // Update derivatives wrt weights
            // Different mode for very first layer
            for (size_t j = 0; j < layer.weights_.cols(); j++) {

                float& W_ij = (del_W_partial_sum[cur_layer_idx])[i][j];
                float delz0_i_del_W_ij = (cur_layer_idx > 0) ? layers_[cur_layer_idx-1].post_activation_vals_[j] : features[j];

                W_ij += imed_value * delz0_i_del_W_ij;
            }
        }
        // The previous gradient is not needed anymore. Set current to previous
        std::swap(current_gradient_, prev_gradient_);
        std::swap(current_jacobian_, prev_jacobian_);
    }
}

std::vector<float> LPP::Network::inference(const std::vector<float>& x) const
{
    return forward_propagation_(
        x, 
        false
    );  // false indicates not training
}
std::vector<float> LPP::Network::inference(const std::span<const float> x) const
{
    std::vector<float> x_v;
    x_v.assign(x.begin(), x.end());

    return forward_propagation_(
        x_v,
        false
    );  // false indicates not training
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
            X_i_                  = std::vector<float>(training_features.cols());
            Y_i_                  = std::vector<float>(training_responses.cols());
    
    // Mini-batch info for SGD
    std::mt19937 shuffler;                                      // Use to shuffle the permutation every epoch
    std::unique_ptr<std::vector<size_t>> permutation = nullptr; // Used to track explanatory variate and response pairs if applicable

    if (options.use_mini_batch()) {
        shuffler = std::mt19937(std::random_device{}());
        mini_batch_size = options.mini_batch_size();

        permutation = std::make_unique<std::vector<size_t>>(num_training_examples);
        std::iota(permutation->begin(), permutation->end(), 0); // permutation = {0, 1, ..., n-1}
    }

    if (options.use_validation()) {
        estimated_validation_responses_ = Matrix(options.validation_responses().rows(), options.validation_responses().cols());
    }

    // How many batches we need to loop through per epoch
    // Add extra batch to train on remainder... this is probably unstable if the remainder is small
    size_t num_batches = num_training_examples / mini_batch_size;
    if (num_training_examples % mini_batch_size != 0) num_batches++;

    // estimated_training_responses: holds predicted values for epoch
    // del_W: stores derivatives wrt to weights
    // del_b: stores derivaiives wrt to biases
    LPP::Matrix                     estimated_training_responses(training_features.rows(), training_responses.cols());
    std::vector<Matrix>             del_W(layers_.size());
    std::vector<std::vector<float>> del_b(layers_.size());
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
            if (options.has_output_stream() && options.use_mini_batch()) {
                options.output_stream() << "\r\tMini-batch " << cur_mini_batch + 1 << " / " << num_batches << std::flush;
            }

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
        float training_data_loss = loss_func_->apply_loss(estimated_training_responses, training_responses);
        // Compute penalty loss
        float penalty_loss = 0.f;
        if (options.use_regularization()) {
            for (auto& layer : layers_) {
                penalty_loss += options.regularizer()->add_regularization_loss_penalty(layer.weights_);
            }
        }
        float training_loss = training_data_loss + penalty_loss;

        // Compute valiation loss if applicable
        // This step will populate the estimated_validation_responses_ field
        float validation_loss;
        if (options.use_validation()) {
            validation_loss = validation_loss_(options.validation_features(), options.validation_responses());
            validation_loss += penalty_loss;
        }

        float training_accuracy, validation_accuracy;
        if (options.performs_classification()) {
            training_accuracy = compute_accuracy(training_responses, estimated_training_responses);

            if (options.use_validation()) {
                validation_accuracy = compute_accuracy(options.validation_responses(), estimated_validation_responses_);
            }
        }

        if (options.has_output_stream()) {
            auto& os = options.output_stream();
            
            if (options.use_mini_batch()) os << '\n';
            os << "\tTraining Loss: " << training_loss << '\n';
            if (options.use_validation()) os << "\tValidation Loss: " << validation_loss << '\n';
            if (options.performs_classification()) os << "\tTraining Accuracy: " << training_accuracy << '\n';
            if (options.performs_classification() && options.use_validation()) os << "\tValidation Accuracy: " << validation_accuracy << '\n';
            os << '\n';
        }
        if (options.has_metadata_stream()) {
            auto& ms = options.metadata_stream();

            ms << cur_epoch + 1 << ' ' << training_loss;
            if (options.use_validation()) ms << ' ' << validation_loss;
            if (options.performs_classification()) ms << ' ' << training_accuracy;
            if (options.performs_classification() && options.use_validation()) ms << ' ' << validation_accuracy;
            ms << '\n';
        }
    }
}

/*
Looping over each training example
Backpropagation will add partial sums to gradients
Have this as a separate function to minimize allocations in training hotpath
*/
void LPP::Network::initialize_gradient_sizes_(
    std::vector<LPP::Matrix>& delL_delW_partial_sum,
    std::vector<std::vector<float>>& delL_delb_partial_sum
) const {
    for (size_t cur_layer_idx = 0; cur_layer_idx < layers_.size(); cur_layer_idx++) {
        size_t in     = layers_[cur_layer_idx].weights_.cols();
        size_t out    = layers_[cur_layer_idx].weights_.rows();

        delL_delW_partial_sum[cur_layer_idx] = Matrix(out, in);
        delL_delb_partial_sum[cur_layer_idx] = std::vector<float>(out, 0.0);
    }
}

/*
Set all the gradient partial sums to 0 for new training cycle
TODO: could spawn separate threads for this
*/
void LPP::Network::initialize_partial_sums_to_zero_(
    std::vector<LPP::Matrix>& delL_delW_partial_sum,
    std::vector<std::vector<float>>& delL_delb_partial_sum
) const {
    for (size_t cur_layer_idx = 0; cur_layer_idx < layers_.size(); cur_layer_idx++) {
        delL_delW_partial_sum[cur_layer_idx].set_all(0.f);
        set_all(delL_delb_partial_sum[cur_layer_idx], 0.f);
    }
}

/*
For all (x_i, y_i) where i in [start, end-1]
- perform forward_propagation_ by firing the pair through the network
- perform backprogagation by moving back through the layers in reverse order
*/
void LPP::Network::process_training_examples_(
    std::vector<LPP::Matrix>& delL_delW,
    std::vector<std::vector<float>>& delL_delb,
    size_t start,
    size_t end,
    const LPP::Matrix& training_features,
    const LPP::Matrix& training_responses,
    LPP::Matrix& estimated_training_responses,
    const std::unique_ptr<std::vector<size_t>>& permutation
) const {
    // X_i_ and Y_i_ have been allocated at the start of train();

    for (size_t t = start; t < end; t++) {
        size_t idx = permutation ? (*permutation)[t] : t;

        X_i_.assign(training_features[idx].begin(), training_features[idx].end());
        Y_i_.assign(training_responses[idx].begin(), training_responses[idx].end());

        estimated_training_responses.set_row(
            idx,
            forward_propagation_(X_i_, true)
        );
        back_propagation_(delL_delW, delL_delb, Y_i_, X_i_);
    }
}

/*
Multiply the derivatives by the learning rate
Divide the derivatives by the batch size because the loss is an average
*/
void LPP::Network::update_parameters_(
    std::vector<Matrix>& delL_delW,
    std::vector<std::vector<float>>& delL_delb,
    size_t batch_size,
    float cur_learning_rate,
    const regular::Regularizer* regularization_option  // shorten this somehow
) {
    for (size_t cur_layer_idx = 0; cur_layer_idx < layers_.size(); cur_layer_idx++) {

        // W <- W - α ∇_W L
        if (regularization_option)
        {
            delL_delW[cur_layer_idx]        *= 1.f / batch_size;
            regularization_option->add_regularization_term_derivative(
                layers_[cur_layer_idx].weights_, delL_delW[cur_layer_idx]);
            
            delL_delW[cur_layer_idx]        *= cur_learning_rate;
            layers_[cur_layer_idx].weights_ -= delL_delW[cur_layer_idx];
        }
        else
        {
            delL_delW[cur_layer_idx]        *= cur_learning_rate / batch_size;
            layers_[cur_layer_idx].weights_ -= delL_delW[cur_layer_idx];
        }
        
        // b <- b - α ∇_b L
        delL_delb[cur_layer_idx]       *= cur_learning_rate / batch_size;
        layers_[cur_layer_idx].biases_ -= delL_delb[cur_layer_idx];

        // TODO: replace these once proven to work
        // scale_then_decrement(layers_[cur_layer_idx].weights_, cur_learning_rate, delL_delW[cur_layer_idx]);
        // scale_then_decrement(layers_[cur_layer_idx].biases_, cur_learning_rate / batch_size, delL_delb[cur_layer_idx]);
    }
}

float LPP::Network::validation_loss_(
    const LPP::Matrix& validation_features,
    const LPP::Matrix& validation_responses
) const {
    // Calculate estimated response
    std::vector<float> X(validation_features.cols());

    for (size_t t = 0; t < validation_responses.rows(); t++)
    {
        X.assign(validation_features[t].begin(), validation_features[t].end());
        estimated_validation_responses_.set_row(
            t,
            forward_propagation_(
                X,
                false /* only final result is needed */
            )
        );
    }

    // Return validation loss
    return loss_func_->apply_loss(estimated_validation_responses_, validation_responses);
}
