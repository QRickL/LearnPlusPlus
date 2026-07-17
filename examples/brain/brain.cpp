#include <fstream>
#include <opencv2/opencv.hpp>
#include "data.hpp"

/*
Data from:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
*/

int main() {

    load_X();
    training_X.normalize();
    
    load_Y();
    validation_X.normalize();

    LPP::Network tumor_classifier = LPP::Network(
        64 * 64,
        {
            {512, &LPP::activations::relu,    nullptr}, // nullptr means weights will be initialized using He initialization
            {256, &LPP::activations::relu,    nullptr},
            {128, &LPP::activations::relu,    nullptr},
            {4,   &LPP::activations::softmax, nullptr}
        }
    );

    // Toggle training settings
    auto training_config = LPP::ExtraTrainingOptions();

    // Minibatch size for stochastic gradient descent
    training_config.set_mini_batch_size(64);

    // Regularization - use L2 for this example
    LPP::regular::ridge.set_lambda(1e-04f);
    training_config.set_regularization(&LPP::regular::ridge);

    // Save training info for visualization later
    std::ofstream ofstr("training_information.txt");

    // Add validation data set
    training_config.set_training_metadata_stream(ofstr);
    training_config.set_validation_data(validation_X, validation_Y);

    // Calculate accuracy during training too
    training_config.set_performs_classification(true);

    // Train!
    tumor_classifier.train(
        training_X,
        training_Y,
        100,
        0.01,
        &LPP::loss::cross_entropy,
        training_config
    );

    // Save model for inference or transfer learning
    tumor_classifier.save_model("tumour_classifier_weights.txt");
    return 0;
}
