#ifndef LPP_TRAINING_OPTION_HPP
#define LPP_TRAINING_OPTION_HPP

#include <cstddef>
#include <iostream>
#include "../regular/regularizer.hpp"

namespace LPP
{

class ExtraTrainingOptions {
    // Mini batch size
    // If this is not null, then we will use gradient descent
    std::unique_ptr<size_t> mini_batch_size_option_;

    // Where should training output be sent?
    std::ostream* training_output_stream_;

    // Where should training metadata be sent?
    std::ostream* training_metadata_stream_;

    // Validation data
    // If validation data lives on the stack, it must not go out of scope before training
    bool use_validation_option_;
    const Matrix* validation_features_;
    const Matrix* validation_responeses_;

    // Regularizer
    regular::Regularizer* regularization_option_;

    // Classification
    bool compute_accuracy_;

public:

ExtraTrainingOptions();
    void set_mini_batch_size(size_t s);
    bool use_mini_batch() const;
    size_t mini_batch_size() const;

    void set_training_output_stream(std::ostream& os);
    bool has_output_stream() const;
    std::ostream& output_stream() const;

    void set_training_metadata_stream(std::ostream& os);
    bool has_metadata_stream() const;
    std::ostream& metadata_stream() const;

    void set_validation_data(const Matrix& validation_features, const Matrix& validation_responses);
    bool use_validation() const;
    const Matrix& validation_features() const;
    const Matrix& validation_responses() const;

    void set_regularization(regular::Regularizer* r);
    bool use_regularization() const;
    regular::Regularizer* regularizer() const;

    void set_performs_classification(bool b);
    bool performs_classification() const;

};
    
} // namespace LPP

#endif