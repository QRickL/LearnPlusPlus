#include "network.h"
#include "check/check.h"
#include <utility>
#include <fstream>
#include <sstream>
#include <iostream>

template <typename number>
LPP::Network<number>::Network(
    size_t inputSize,
    const std::vector<LayerInfo>& layerInfos,
    ProbDistnPtr probDistn
) {
    _lpp_check_(!layer_info.empty(), "Attempted to construct network with no layrs");

    loss_func__ = nullptr;
    size_t cur_in_size__;
    size_t cur_out_size__ = inputSize;

    for (const LayerInfo& layer : layerInfos) {
        cur_in_size__  = cur_out_size__;
        cur_out_size__ = layer.first;

        layers__.push_back(std::make_unique<Layer<number>>(cur_in_size__, cur_out_size__, layer.second, probDistn));
    }
}

// TODO: implement this back last
// LPP::Network::Network(const std::string& filepath)
// {
//     std::ifstream model_file{filepath};
//     if (!model_file.is_open()) {
//         const auto msg = "Failed to read model from file: " + filepath;
//         throw std::ios_base::failure(msg);
//     }
//     std::cout << "\nLoading neural network from file: " + filepath << '\n';
//     std::string cur_line;

//     // Verify we are reading a correct file format
//     getline(model_file, cur_line);
//     if (cur_line != "LearnPlusPlus") {
//         const auto msg = "Not given a LearnPlusPlus file";
//         throw std::ios_base::failure(msg);
//     }
//     getline(model_file, cur_line);
//     if (cur_line != "NeuralNetwork") {
//         const auto msg = "File type is not NeuralNetwork";
//         throw std::ios_base::failure(msg);
//     }

//     loss_func__ = nullptr;
//     while (true) {
//         // Read in layer number
//         model_file >> cur_line;
//         if (cur_line == "END") break;
        
//         double val;
//         size_t out, in;
//         model_file >> out >> in;
//         std::cout << cur_line << ":\n";
//         std::cout << "Input size: " << in << "\nOutput size: " << out << '\n';

//         // Read in weights__
//         auto cur_weights = std::make_unique<Matrix>(out, in);
//         for (size_t i = 0; i < out; i++) {
//             for (size_t j = 0; j < in; j++) {
//                 model_file >> val;
//                 cur_weights->set(i,j,val);
//             }
//         }
//         // Read in biases__
//         auto cur_biases = std::make_unique<std::vector<double>>(out);
//         for (size_t i = 0; i < out; i++) {
//             model_file >> val;
//             (*cur_biases)[i] = val;
//         }

//         // Read in activation
//         model_file >> cur_line;
//         std::shared_ptr<Activation> cur_act;
//         try {
//             cur_act = LPP::chooseActivation.at(cur_line);
//         } catch (std::out_of_range oor) {
//             const auto msg = "Invalid activation function: " + cur_line;
//             throw std::ios_base::failure(msg);
//         }
//         std::cout << "Activation: " + cur_line << "\n\n";

//         // Construct layer and push back
//         layers__.push_back(std::make_unique<Layer>(cur_weights, cur_biases, cur_act));
//     }
//     std::cout << std::endl;
// }

// TODO: reimplement
// void LPP::Network::saveModel(const std::string& filepath) const
// {
//     std::ofstream model_file{filepath};
//     if (!model_file.is_open()) {
//         const auto msg = "Failed to save model to file: " + filepath;
//         throw std::ios_base::failure(msg);
//     }
//     model_file << "LearnPlusPlus\nNeuralNetwork\n";

//     for (size_t cur_l = 0; cur_l < layers__.size(); cur_l++) {
//         // What layer?
//         model_file << "Layer" << cur_l << '\n';

//         // Matrix dimensions
//         const size_t out = layers__[cur_l]->weights__->rows();
//         const size_t in = layers__[cur_l]->weights__->cols();
//         model_file << out << ' ' << in << '\n';

//         // Weight contents
//         for (size_t i = 0; i < out; i++) {
//             for (size_t j = 0; j < in; j++) {
//                 model_file << (*layers__[cur_l]->weights__)[i][j] << ' ';
//             }
//             model_file << '\n';
//         }

//         // Bias contents
//         for (size_t i = 0; i < layers__[cur_l]->biases__->size(); i++) {
//             model_file << (*layers__[cur_l]->biases__)[i] << ' ';
//         }
//         model_file << '\n';

//         // Activation function
//         model_file << layers__[cur_l]->activation_func__->who() << '\n';
//     }
//     model_file << "END\n";
//     model_file << LPP::model_save_end_msg__ << '\n';
//     model_file.close();
// }

template <typename number>
LPP::Vect<number> LPP::Network<number>::forward_propagation__(Vect<number> current_fire__, bool training__) const
{
    for (auto& layer : layers__) {
        // z = Wx + b
        current_fire__ = (*layer->weights__) * current_fire__ + (*layer->biases__);
        if (training__) {
            // Store z
            layer->pre_activation__ = current_fire__;
        }

        // a = σ(z)
        layer->apply_activation_layer__(current_fire__);
        if (training__) {
            // Store a
            layer->post_activation__ = current_fire__;
        }
    }
    return current_fire__;
}

template <typename number>
void LPP::Network<number>::back_propagation__(
    std::vector<Weights>&       del_W_partial_sum__,
    std::vector<Biases>&        del_b_partial_sum__,
    const std::vector<double>&  response_var__, // these have to be converted to vect number
    const std::vector<double>&  explan_var__
) const {
    // // std::vector<std::unique_ptr<Matrix>>& del_W_partial_sum, std::vector<std::unique_ptr<std::vector<double>>>& del_b_partial_sum, const std::vector<double>& response_var, const std::vector<double>& explan_var) const
    
    // Used to calculate derivatives recursively
    Vect<number> __prev_gradient;
    Vect<number> __current_gradient;

    // Don't use size_t to avoid underflow
    for (int cur_l = layers__.size() - 1; cur_l >=0 ; cur_l--) {
        
        if (cur_l == layers__.size() - 1) {
            // Looking at last layer, calculate gradient using loss
            __current_gradient = loss_func__->findGradient(layers__[cur_l]->post_activation__, response_var);
        }
        else {
            // Looking at non-last layer, calculate gradient recursively
            size_t __forward_layer_size = layers__[cur_l+1]->weights__->rows();
            size_t __current_layer_size = layers__[cur_l]->weights__->rows();
            __current_gradient = Vect<number>(__current_layer_size, 0.0);

            for (size_t c = 0; c < __current_layer_size; c++) {
                for (size_t k = 0; k < __forward_layer_size; k++) {

                    number delL_dela1 = __prev_gradient[k];
                    number dela1_delz = layers__[cur_l+1]->activation_func__->applyDerivative(layers__[cur_l+1]->pre_activation__[k]);
                    number delz_dela0 = layers__[cur_l+1]->weights__->get(k,c);

                    // Chain rule: delL_dela0 = delL_dela1 * dela1_delz * delz_dela0
                    __current_gradient[c] += delL_dela1 * dela1_delz * delz_dela0;
                }
            }
        }

        for (size_t i = 0; i < layers__[cur_l]->weights__->rows(); i++) {

            // Store this for one less multiplication, but mainly because its unreadable
            number __imed_val = __current_gradient[i] * layers__[cur_l]->activation_func__->applyDerivative(layers__[cur_l]->pre_activation__[i]);

            // Update derivatives wrt bias
            (*del_b_partial_sum__[cur_l])[i] += __imed_val;
            
            // Update derivatives wrt weights
            for (size_t j = 0; j < layers__[cur_l]->weights__->cols(); j++) {
                if (cur_l > 0) {
                    (*del_W_partial_sum__[cur_l])[i][j] += __imed_val * layers__[cur_l-1]->post_activation__[j];
                } else {
                    // Executes on the first layer
                    (*del_W_partial_sum__[cur_l])[i][j] += __imed_val * explan_var[j];
                }
            }
        }
        // TODO: see if this breaks anything
        std::swap(__current_gradient, __prev_gradient);
    }
}

std::vector<double> LPP::Network::inference(const std::vector<double>& x) const
{
    // convert to vect
    return forward_propagation__(x, false);
    // covert to std::vect
}

std::vector<float> inference(const Vect<number>& x) const; // TODO: need to implements this one

double LPP::Network::train(
    const Matrix& explan_var,
    const Matrix& response_var,
    const size_t epochs,
    const double init_learning_rate,
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
    double learning_rate = init_learning_rate;
    loss_func__ = loss_ptr;
    double loss;

    // response_var_hat: holds predicted values for epoch
    // del_W: stores derivatives wrt to weights
    // del_b: stores derivaiives wrt to biases
    LPP::Matrix                                         response_var_hat(explan_var.rows(), explan_var.cols());
    std::vector<std::unique_ptr<Matrix>>                del_W(layers__.size());
    std::vector<std::unique_ptr<std::vector<double>>>   del_b(layers__.size());

    for (size_t cur_epoch = 0; cur_epoch < epochs; cur_epoch++) {
        std::cout << "Training epoch " << cur_epoch + 1 << ":\n";

        // Initialize all gradient sums to 0
        for (size_t cur_l = 0; cur_l < layers__.size(); cur_l++) {
            const size_t in     = layers__[cur_l]->weights__->cols();
            const size_t out    = layers__[cur_l]->weights__->rows();

            del_W[cur_l] = std::make_unique<Matrix>(out, in);
            del_b[cur_l] = std::make_unique<std::vector<double>>(out, 0.0);
        }
        // Looping over each training example
        // Backpropagation will add partial sums to gradients
        for (size_t t = 0; t < num_training_examples; t++) {

            response_var_hat[t] = forward_propagation__(explan_var[t], true);
            back_propagation__(del_W, del_b, response_var[t], explan_var[t]);

        }
        // Update parameters!
        for (size_t cur_l = 0; cur_l < layers__.size(); cur_l++) {
            // W <- W - α ∇_W L
            *del_W[cur_l]           *= learning_rate / num_training_examples;
            *layers__[cur_l]->weights__ -= *del_W[cur_l];

            // b <- b - α ∇_b L
            *del_b[cur_l]           *= learning_rate / num_training_examples;
            *layers__[cur_l]->biases__  -= *del_b[cur_l];
        }
        // Compute loss
        loss = loss_func__->apply_itself(response_var_hat, response_var);
        std::cout << "Loss: " << loss << "\n\n";
    }
    return loss;
}
