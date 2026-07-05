#ifndef LPP_TRAINING_OPTION_HPP
#define LPP_TRAINING_OPTION_HPP

#include <cstddef>
#include "../regular/regularizer.hpp"
#include <iostream>
#include "../checking/check.hpp"

namespace LPP
{

class Network;

class ExtraTrainingOptions {

    // Mini batch size
    // If this is not null, then we will use gradient descent
    std::unique_ptr<size_t> mini_batch_size_option_;

    // Where should training output be sent?
    std::ostream* training_output_stream_;

    // Where should training metadata be sent?
    std::ostream* training_metadata_stream_;

    // Validation data
    bool use_validation_option_;
    const Matrix* validation_features_option_;
    const Matrix* validation_responeses_option_;

    // std::unique_ptr<regular::Regularizer> regularization_option_;

public:

ExtraTrainingOptions() :
    mini_batch_size_option_{nullptr},
    training_output_stream_{&std::cout},
    training_metadata_stream_{nullptr},
    use_validation_option_{false},
    validation_features_option_{nullptr},
    validation_responeses_option_{nullptr}
{}

void set_mini_batch_size(size_t s) {
    mini_batch_size_option_ = std::make_unique<size_t>(s);
}

void set_training_output_stream(std::ostream& os) {
    training_output_stream_ = &os;
}

void set_training_metadata_stream(std::ostream& os) {
    training_metadata_stream_ = &os;
}

void set_validation_data(
    const Matrix& validation_features,
    const Matrix& validation_responses
) {
    enforce_condition(validation_features.rows() == validation_responses.rows(),
        "ExtraTrainingOptions::set_validation_data - validaation features and reponses have different number of rows");

    validation_features_option_ = &validation_features;
    validation_responeses_option_ = &validation_responses; // TODO: check if this is like... bad for me
    use_validation_option_ = true;
}


// void set_regularization(regular::Regularizer r) {
//     regularization_option_ = std::make_unique<regular::Regularizer>(r);
// }


bool use_mini_batch() const { return mini_batch_size_option_ != nullptr; }
size_t mini_batch_size() const { return *mini_batch_size_option_; }

bool has_output_stream() const { return training_output_stream_ != nullptr; }
std::ostream& output_stream() const { return *training_output_stream_; }

bool has_metadata_stream() const { return training_metadata_stream_ != nullptr; }
std::ostream& metadata_stream() const { return *training_metadata_stream_; }

bool use_validation() const { return use_validation_option_; }
const Matrix& validation_features() const { return *validation_features_option_; }
const Matrix& validation_responses() const { return *validation_responeses_option_; }

// const regular::Regularizer* regularizer() const { return regularization_option_; }


};
    
} // namespace LPP

#endif