#include "cmath"

// To use identity function, pass in nullptr to layer constructor
// Custom activations can be used, just pass in the ptr for a double -> double function
// Hyberbolic tanget already defined in cmath

// Sigmoid
inline double sigmoid(double x) {
    return 1/(1+exp(-x));
}

// Rectified linear unit
inline double ReLU(double x) {
    return (x > 0) ? x : 0;
}
