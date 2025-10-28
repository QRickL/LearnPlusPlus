#ifndef LPP_NETWORK_H
#define LPP_NETWORK_H

#include "matrix.h"
#include "functions/losses.h"
#include <fstream>
#include <sstream>

namespace LPP {

const std::string model_save_end_msg = "END\nCheck out: https://github.com/QRickL/LearnPlusPlus\n";

class Network {
public:
    // TODO: compile
    // TODO: training
    // TODO: add option to set optimizers etc...
    // TODO: activation will be given to layers to choose
    // TODO: initial learning rate
    // TODO: early stopping
    // TODO: loss functions (start with mse and entropy)
    // TODO: configure optimizers

    // Network is abstract
    virtual ~Network() = 0;

    virtual void save_model(const std::string& filepath) const = 0;
    
    virtual std::vector<double> inference(const std::vector<double>& x) const = 0;
};

} // namespace LPP

#endif