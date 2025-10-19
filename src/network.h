#ifndef LPP_NETWORK_H
#define LPP_NETWORK_H

#include "matrix.h"

namespace LPP {

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
};

} // namespace LPP

#endif