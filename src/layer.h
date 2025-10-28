#ifndef LPP_LAYER_H
#define LPP_LAYER_H

#include "functions/activations.h"

namespace LPP {

class Layer {
public:
    // Layer is abstract
    virtual ~Layer() = 0;
};

} // namespace LPP

#endif