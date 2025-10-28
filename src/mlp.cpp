#include "mlp.h"

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

// Could I give some of this to network instead????
LPP::MLP::MLP(const size_t input_size, const std::vector<std::pair<size_t, std::shared_ptr<Activation>>> layer_info)
{
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

// TO DO: fill in
std::vector<double> LPP::MLP::forward_propagation(const std::vector<double>& x, const bool saving) const
{
    std::vector<double> current_fire = x;
    for (auto& layer : layers) {
        current_fire = (*layer->weights) * current_fire + (*layer->biases);
        layer->apply_activation(current_fire);

        // Reduce overhead associated with saving if not needed
        if (saving) layer->intermed_val = current_fire;
    }
    return current_fire;
}

std::vector<double> LPP::MLP::inference(const std::vector<double>& layer_info) const
{
    return forward_propagation(layer_info, false);
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
        // Use .at instead of [] because [] is not const
        model_file << layers[l]->act_func->who() << '\n';
    }
    model_file << LPP::model_save_end_msg;
    model_file.close();
}
