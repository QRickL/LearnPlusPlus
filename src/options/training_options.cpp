#include "training_options.hpp"
#include "../checking/check.hpp"

LPP::ExtraTrainingOptions::ExtraTrainingOptions() :
    mini_batch_size_option_{nullptr},
    training_output_stream_{&std::cout},
    training_metadata_stream_{nullptr},
    use_validation_option_{false},
    validation_features_{nullptr},
    validation_responeses_{nullptr}
{}

void LPP::ExtraTrainingOptions::set_mini_batch_size(size_t s) {
    enforce_condition(s != 0,
        "ExtraTrainingOptions::set_mini_batch_size - batch size cannot be 0");

    mini_batch_size_option_ = std::make_unique<size_t>(s);
}

void LPP::ExtraTrainingOptions::set_training_output_stream(std::ostream& os) {
    training_output_stream_ = &os;
}

void LPP::ExtraTrainingOptions::set_training_metadata_stream(std::ostream& os) {
    training_metadata_stream_ = &os;
}

void LPP::ExtraTrainingOptions::set_validation_data(
    const Matrix& validation_features,
    const Matrix& validation_responses
) {
    enforce_condition(validation_features.rows() == validation_responses.rows(),
        "ExtraTrainingOptions::set_validation_data - validaation features and reponses have different number of rows");

    validation_features_ = &validation_features;
    validation_responeses_ = &validation_responses; // TODO: check if this is like... bad for me
    use_validation_option_ = true;
}


void LPP::ExtraTrainingOptions::set_regularization(const regular::Regularizer* r) {
    regularization_option_ = r;
}


bool LPP::ExtraTrainingOptions::use_mini_batch() const { return mini_batch_size_option_ != nullptr; }
size_t LPP::ExtraTrainingOptions::mini_batch_size() const { return *mini_batch_size_option_; }

bool LPP::ExtraTrainingOptions::has_output_stream() const { return training_output_stream_ != nullptr; }
std::ostream& LPP::ExtraTrainingOptions::output_stream() const { return *training_output_stream_; }

bool LPP::ExtraTrainingOptions::has_metadata_stream() const { return training_metadata_stream_ != nullptr; }
std::ostream& LPP::ExtraTrainingOptions::metadata_stream() const { return *training_metadata_stream_; }

bool LPP::ExtraTrainingOptions::use_validation() const { return use_validation_option_; }
const LPP::Matrix& LPP::ExtraTrainingOptions::validation_features() const { return *validation_features_; }
const LPP::Matrix& LPP::ExtraTrainingOptions::validation_responses() const { return *validation_responeses_; }

bool LPP::ExtraTrainingOptions::use_regularization() const { return regularization_option_ != nullptr; }
const LPP::regular::Regularizer* LPP::ExtraTrainingOptions::regularizer() const { return regularization_option_; }